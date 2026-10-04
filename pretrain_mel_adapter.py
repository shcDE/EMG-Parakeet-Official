import os
import sys
import math
import logging
import subprocess

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn.functional as F
import tqdm

from read_emg import EMGDataset

from architecture import Model
from data_utils import phoneme_inventory
from mel_pipeline import predict_mel_single
# the shared flags (start_training_from / output_directory etc.) are defined here
import transduction_model  # noqa: F401
from parakeet_mel_asr import ParakeetMelASR

from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('pretrain_epochs', 10, 'Number of MelAdapter warmup epochs')
flags.DEFINE_float('pretrain_lr', 1e-3, 'MelAdapter warmup learning rate')
flags.DEFINE_integer('pretrain_batch_size', 8, 'Warmup batch size (utterances)')
flags.DEFINE_integer('pretrain_max_utts', 0, 'Max utterances to use (0 = all)')
flags.DEFINE_boolean('compare_stats_only', False,
                     'Skip training; only compare adapter output vs. real feature '
                     'statistics, then exit')
flags.DEFINE_boolean('length_mse_only', False,
                     'Skip training; measure MSE bucketed by utterance length, then exit. '
                     'Used to determine whether interpolation (F.interpolate) is the '
                     'bottleneck. Only meaningful with a warmed-up adapter supplied via '
                     '--parakeet_checkpoint.')
flags.DEFINE_integer('length_mse_max_utts', 1000,
                     'Max utterances used for the length-bucketed MSE measurement')
flags.DEFINE_boolean('pretrain_use_target_length', True,
                     'During warmup, use the real preprocessor output length instead of the '
                     'formula (mel_len*100/86.13). When the two lengths disagree, '
                     'align_target forcibly compresses the time axis and contaminates the '
                     'MSE. At inference there is no target, so the formula is used; for '
                     'normal utterances the ratio is around 0.99, so the mismatch is '
                     'negligible.')
flags.DEFINE_float('pretrain_max_ratio_dev', 1.0,
                   'Exclude utterances from warmup when the formula length / real length '
                   'ratio deviates from 1 by at least this much (suspected parallel-audio '
                   'mismatch). 0 disables the exclusion.')


# ---------------------------------------------------------------- helpers

def load_target_features(asr, audio_path, device):
    """wav -> Parakeet preprocessor -> (feat_dim, T) real features"""
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    wav = torch.from_numpy(audio).unsqueeze(0).to(device)
    wav_len = torch.tensor([wav.shape[1]], device=device)
    with torch.no_grad():
        feats, feats_len = asr.asr.preprocessor(input_signal=wav, length=wav_len)
    return feats[0, :, :int(feats_len[0])]          # (feat_dim, T)


def collect_pairs(dataset, transduction_model, asr, device, indices):
    """Collect (predicted mel, real feature) pairs. Utterances that fail are skipped."""
    mels, targets = [], []
    for i in indices:
        dp = dataset[i]
        path = dp.get('audio_file')
        if not path or not os.path.exists(path):
            continue
        try:
            tgt = load_target_features(asr, path, device)
        except Exception as e:  # noqa: BLE001
            logging.debug('Failed to load audio %s: %s', path, e)
            continue
        mel = predict_mel_single(transduction_model, dp, device)
        mels.append(mel.detach())
        targets.append(tgt)
    return mels, targets


def align_target(tgt, length):
    """(feat_dim, T_true) -> interpolate the time axis to (feat_dim, length)."""
    if tgt.shape[1] == length:
        return tgt
    return F.interpolate(tgt.unsqueeze(0), size=length,
                         mode='linear', align_corners=False).squeeze(0)


def describe(name, x):
    """Summarize the distribution of a (D, T) tensor.

    Because of per_feature normalization, mean/std are forced to 0/1 on both
    sides, so the real difference shows up in the shape of the distribution
    (skew, floor at the lower end). A real log-mel is an asymmetric distribution
    with silent regions piled up at the floor.
    """
    f = x.flatten().float()
    q = torch.quantile(f, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=f.device))
    skew = float(((f - f.mean()) ** 3).mean() / (f.std() ** 3 + 1e-8))
    logging.info('%-16s shape=%s  mean=%+.3f std=%.3f  min=%+.2f max=%+.2f  '
                 'p1=%+.2f p5=%+.2f p50=%+.2f p95=%+.2f p99=%+.2f  skew=%+.2f',
                 name, tuple(x.shape), float(f.mean()), float(f.std()),
                 float(f.min()), float(f.max()),
                 *[float(v) for v in q], skew)


def length_mse_report(asr, transduction_model, dataset, device, max_utts):
    """Measure MSE bucketed by utterance length to decide whether interpolation is
    the bottleneck.

    The MelAdapter matches the 86.13 Hz -> 100 Hz time axis with
    F.interpolate(mode='linear'), which is a fixed, non-learnable operation. The
    longer the utterance, the more frames interpolation has to invent, so if MSE
    clearly rises with length, interpolation is the bottleneck. If instead it stays
    flat regardless of length, the bottleneck lies elsewhere (e.g. a mismatch in the
    frequency-axis definition).
    """
    n = min(max_utts, len(dataset))
    records = []

    asr.mel_adapter.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(n), 'Length-MSE', disable=None):
            mels, tgts = collect_pairs(dataset, transduction_model, asr, device, [i])
            if not mels:
                continue
            mel, tgt = mels[0], tgts[0]
            L = asr.num_feature_frames(mel.shape[0])
            out = asr.mel_adapter([mel], [L])[0]              # (D, L)
            mse = float(F.mse_loss(out, align_target(tgt, L)))
            # Interpolation ratio: frames invented relative to the real frame count
            ratio = L / max(1, tgt.shape[1])
            records.append((mel.shape[0], L, tgt.shape[1], ratio, mse))

    if not records:
        logging.error('There are no utterances to measure.')
        return

    arr = np.array(records, dtype=np.float64)
    mel_len, feat_len, true_len, ratio, mse = arr.T

    logging.info('=' * 88)
    logging.info('MSE bucketed by utterance length  (n=%d)', len(records))
    logging.info('=' * 88)

    edges = [0, 200, 400, 600, 800, 1000, 1400, 10 ** 9]
    labels = ['<200', '200-400', '400-600', '600-800', '800-1000', '1000-1400', '1400+']
    logging.info('%-12s %6s %10s %10s %10s %10s',
                 'mel frames', 'n', 'MSE mean', 'MSE med', 'MSE std', 'interp ratio')
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        m = (mel_len >= lo) & (mel_len < hi)
        if m.sum() == 0:
            continue
        logging.info('%-12s %6d %10.5f %10.5f %10.5f %10.4f',
                     lab, int(m.sum()), mse[m].mean(), np.median(mse[m]),
                     mse[m].std(), ratio[m].mean())

    # Correlation: do length and MSE actually move together?
    if len(records) > 2:
        r_len = float(np.corrcoef(mel_len, mse)[0, 1])
        r_ratio = float(np.corrcoef(ratio, mse)[0, 1])
        logging.info('-' * 88)
        logging.info('corr(mel length, MSE)     = %+.4f', r_len)
        logging.info('corr(interp ratio, MSE)   = %+.4f', r_ratio)
        logging.info('overall MSE mean = %.5f, median = %.5f', mse.mean(), np.median(mse))
        logging.info('-' * 88)
        if r_len > 0.3:
            logging.info('Verdict: clear positive correlation between length and MSE -> '
                         'interpolation / long-range context is the bottleneck. Replace '
                         'F.interpolate with a learnable time-axis transform.')
        elif r_len < 0.1:
            logging.info('Verdict: length and MSE are essentially unrelated -> interpolation '
                         'is not the bottleneck. More likely a structural floor such as a '
                         'frequency-axis definition mismatch (22.05 kHz mel vs 16 kHz '
                         'filterbank). Enlarging the adapter will not help.')
        else:
            logging.info('Verdict: weak correlation. Interpolation contributes somewhat but '
                         'is not the main bottleneck.')

    # Details for the 5 best and 5 worst utterances
    order = np.argsort(mse)
    logging.info('5 lowest MSE (mel frames, feat frames, real frames, ratio, MSE):')
    for j in order[:5]:
        logging.info('  %5d %5d %5d %.4f %.5f', *arr[j])
    logging.info('5 highest MSE:')
    for j in order[-5:]:
        logging.info('  %5d %5d %5d %.4f %.5f', *arr[j])
    logging.info('=' * 88)


# ------------------------------------------------------------------- main

def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(handlers=[
        logging.FileHandler(os.path.join(FLAGS.output_directory, 'pretrain_log.txt'), 'w'),
        logging.StreamHandler()
    ], level=logging.INFO, format="%(message)s")
    logging.info(sys.argv)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainset = EMGDataset(dev=False, test=False)
    devset = EMGDataset(dev=True)
    logging.info('train / dev split: %d %d', len(trainset), len(devset))

    transduction_model = Model(devset.num_features, devset.num_speech_features,
                               len(phoneme_inventory)).to(device)
    assert FLAGS.start_training_from is not None, \
        'Specify the stage-1 transduction model (.pt) with --start_training_from.'
    transduction_model.load_state_dict(torch.load(FLAGS.start_training_from), strict=False)
    transduction_model.eval()
    for p in transduction_model.parameters():
        p.requires_grad = False

    asr = ParakeetMelASR(devset.num_speech_features, device=device)
    if FLAGS.parakeet_checkpoint is not None:
        asr.load(FLAGS.parakeet_checkpoint)
    asr.eval()   # the encoder/decoder are unused, but this turns dropout off

    # ---------------- Distribution comparison (diagnostic) ----------------
    logging.info('=' * 78)
    logging.info('MelAdapter output vs. real preprocessor feature distribution')
    logging.info('=' * 78)
    with torch.no_grad():
        mels, tgts = collect_pairs(trainset, transduction_model, asr, device, range(3))
        for k, (mel, tgt) in enumerate(zip(mels, tgts)):
            L = asr.num_feature_frames(mel.shape[0])
            out = asr.mel_adapter([mel], [L])[0]
            logging.info('--- utterance %d (mel %d frames -> feat %d frames, real %d frames) ---',
                         k, mel.shape[0], L, tgt.shape[1])
            describe('adapter output', out)
            describe('real feature', align_target(tgt, L))
    logging.info('=' * 78)

    if FLAGS.compare_stats_only:
        return

    if FLAGS.length_mse_only:
        if FLAGS.parakeet_checkpoint is None:
            logging.warning('Measuring without --parakeet_checkpoint means the adapter is '
                            'random, so length dependence cannot be determined.')
        length_mse_report(asr, transduction_model, trainset, device,
                          FLAGS.length_mse_max_utts)
        return

    # ---------------- Warmup training ----------------
    n = len(trainset) if FLAGS.pretrain_max_utts <= 0 else min(FLAGS.pretrain_max_utts,
                                                               len(trainset))
    indices = list(range(n))
    optim = torch.optim.AdamW(asr.mel_adapter.parameters(), lr=FLAGS.pretrain_lr,
                              betas=(0.9, 0.98), weight_decay=0.0)
    bs = FLAGS.pretrain_batch_size
    total_steps = max(1, FLAGS.pretrain_epochs * math.ceil(n / bs))
    step = 0

    best = float('inf')
    n_skipped = 0
    for epoch in range(FLAGS.pretrain_epochs):
        np.random.shuffle(indices)
        asr.mel_adapter.train()
        losses = []

        for s in tqdm.tqdm(range(0, n, bs), f'Pretrain ep{epoch+1}', disable=None):
            chunk = indices[s:s + bs]
            with torch.no_grad():
                mels, tgts = collect_pairs(trainset, transduction_model, asr, device, chunk)
            if not mels:
                continue

            # Exclude utterances whose formula length / real length ratio is badly off.
            # (A duration mismatch between the predicted mel and the parallel audible
            #  audio breaks the time-axis correspondence and produces outliers with
            #  MSE above 1.0.)
            if FLAGS.pretrain_max_ratio_dev > 0:
                keep_m, keep_t = [], []
                for m, t in zip(mels, tgts):
                    ratio = asr.num_feature_frames(m.shape[0]) / max(1, t.shape[1])
                    if abs(ratio - 1.0) <= FLAGS.pretrain_max_ratio_dev:
                        keep_m.append(m)
                        keep_t.append(t)
                    else:
                        n_skipped += 1
                mels, tgts = keep_m, keep_t
                if not mels:
                    continue

            if FLAGS.pretrain_use_target_length:
                # Use the real feature length directly, so align_target becomes the
                # identity and only the pure frequency mapping is learned, without
                # time-axis distortion
                lens = [t.shape[1] for t in tgts]
            else:
                lens = [asr.num_feature_frames(m.shape[0]) for m in mels]

            feats = asr.mel_adapter(mels, lens)              # (B, 80, T_max)

            loss = 0.0
            for i, (L, tgt) in enumerate(zip(lens, tgts)):
                loss = loss + F.mse_loss(feats[i, :, :L], align_target(tgt, L))
            loss = loss / len(mels)

            # cosine decay
            prog = step / total_steps
            for g in optim.param_groups:
                g['lr'] = FLAGS.pretrain_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(asr.mel_adapter.parameters(), 1.0)
            optim.step()
            losses.append(float(loss))
            step += 1

        mean_loss = float(np.mean(losses)) if losses else float('nan')
        logging.info('epoch %d - feature MSE: %.5f', epoch + 1, mean_loss)

        if mean_loss < best:
            best = mean_loss
            asr.save(os.path.join(FLAGS.output_directory, 'parakeet_asr_pretrained.pt'))

    logging.info('best feature MSE: %.5f', best)
    if n_skipped:
        logging.info('Utterance-epochs excluded for ratio anomalies: %d '
                     '(about %.1f per epoch, %.1f%% of the total)',
                     n_skipped, n_skipped / max(1, FLAGS.pretrain_epochs),
                     100.0 * n_skipped / max(1, FLAGS.pretrain_epochs * n))
    logging.info('Next: train_parakeet.py --parakeet_checkpoint %s',
                 os.path.join(FLAGS.output_directory, 'parakeet_asr_pretrained.pt'))

    # ---------------- Distribution comparison after warmup ----------------
    logging.info('=' * 78)
    logging.info('Distribution after warmup')
    logging.info('=' * 78)
    asr.mel_adapter.eval()
    with torch.no_grad():
        mels, tgts = collect_pairs(trainset, transduction_model, asr, device, range(3))
        for k, (mel, tgt) in enumerate(zip(mels, tgts)):
            L = asr.num_feature_frames(mel.shape[0])
            out = asr.mel_adapter([mel], [L])[0]
            logging.info('--- utterance %d ---', k)
            describe('adapter output', out)
            describe('real feature', align_target(tgt, L))


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()