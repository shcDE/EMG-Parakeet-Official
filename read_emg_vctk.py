import os
import re
import sys
import json
import copy
import string
import random
import pickle
import logging
from functools import lru_cache

import numpy as np
import scipy.signal
import soundfile as sf
import librosa
import torch
import jiwer
from unidecode import unidecode

from data_utils import (get_emg_features, FeatureNormalizer, phoneme_inventory,
                        TextTransform, mel_spectrogram)

from absl import flags

FLAGS = flags.FLAGS



flags.DEFINE_string('manifest_file', '/home/shcheong24/Desktop/experiment_shc/first_author/speech_synthesis_emg/experiment/STT_optimized/emg_vctk_manifest.json',
               'EMG-VCTK manifest (output of emg_vctk_index.py)')
flags.DEFINE_integer('vctk_emg_sampling_rate', 800, 'Original EMG sampling rate (Hz)')
flags.DEFINE_integer('vctk_mains_freq', 50, 'Mains noise frequency (Germany = 50 Hz)')
flags.DEFINE_integer('vctk_num_channels', 8, 'Number of EMG channels')
flags.DEFINE_list('vctk_remove_channels', [], 'Indices of channels to zero out')
flags.DEFINE_bool('vctk_silent_only', False,
                  'Use only silent utterances in the training split. When False, audible '
                  'utterances are included as well, which keeps the direct cross-entropy '
                  'path in dtw_loss alive (recommended)')
flags.DEFINE_bool('vctk_eval_silent_only', True,
                  'Build dev/test from silent utterances only. The paper evaluates on '
                  'silent test EMG only (Section 3.3, Table 2), so keep the default of True')
flags.DEFINE_bool('vctk_noisereduce', False,
                  'Apply noisereduce (non-stationary) to the target audio (paper Section 3.1)')
flags.DEFINE_float('vctk_raw_emg_divisor', 0.0,
                   'If > 0, divide raw EMG by this value. If 0, use the scale estimated '
                   'by the normalizer')

# 22050 / hop 256 = 86.1328125 fps (keeps the Gaddy / BigVGAN setup)
FRAME_RATE = 22050 / 256
EMG_FEATURE_SR = FRAME_RATE * 6      # 516.797 Hz -> get_emg_features hop=6
EMG_RAW_SR = FRAME_RATE * 8          # 689.062 Hz -> ResBlock stride 2^3 = 8
FEATURE_OFFSET = 6
RAW_OFFSET = 8


# ===========================================================================
# Text preprocessing (identical to ParakeetDatasetCreator in create_parakeet_dataset.py)
# ---------------------------------------------------------------------------
# If the normalization used to build the Parakeet manifest differs from the
# label normalization here, the WER computation is measured against a different
# reference, so both use exactly the same pipeline:
#   unidecode -> replace '-' / ':' with spaces -> RemovePunctuation + ToLowerCase
#   -> strip [^a-zA-Z0-9\s] -> collapse repeated whitespace
#
# If preprocessing yields an empty string, the utterance is kept and a
# placeholder is inserted instead of dropping it. The "skip utterances with no
# characters" rule in SizeAwareSampler has also been removed, so utterances with
# an empty transcript go into the training batches as-is.
#
# Note 1) data_utils.TextTransform.chars does not contain '.', and clean_text
#         runs RemovePunctuation, so text_to_int('.') still returns an empty
#         array. Stage 1 does not use text in its loss, so training is unaffected.
# Note 2) A placeholder is not real utterance content, so if it enters the WER
#         reference the score for that utterance is meaningless. Filter these out
#         at evaluation time with is_placeholder().
# ===========================================================================

EMPTY_TEXT_PLACEHOLDER = '.'

_TEXT_TRANSFORMATION = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])


@lru_cache(maxsize=None)
def preprocess_text(text: str) -> str:
    if not text:
        return EMPTY_TEXT_PLACEHOLDER

    text = unidecode(text)

    text = text.replace('-', ' ')
    text = text.replace(':', ' ')

    text = _TEXT_TRANSFORMATION(text)

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    if not text:
        return EMPTY_TEXT_PLACEHOLDER

    return text


def is_placeholder(text: str) -> bool:
    """Check whether the transcript was empty and replaced by the placeholder
    (used to exclude the utterance from evaluation)."""
    return text.strip() == EMPTY_TEXT_PLACEHOLDER


# ===========================================================================
# Signal preprocessing
# ===========================================================================

def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)


def notch_harmonics(signal, freq, sample_frequency, max_harmonic=7):
    """Skip harmonics above Nyquist (required at an 800 Hz sampling rate).

    Using 60 Hz directly would put the 7th harmonic at 420 Hz > Nyquist (400 Hz),
    which makes iirnotch raise an error. 50 Hz hits the same problem from the 8th
    harmonic (400 Hz), so the guard is applied in both cases.
    """
    nyq = sample_frequency / 2.0
    for harmonic in range(1, max_harmonic + 1):
        f = freq * harmonic
        if f >= nyq * 0.98:
            break
        signal = notch(signal, f, sample_frequency)
    return signal


def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal)) / old_freq
    sample_times = np.arange(0, times[-1], 1 / new_freq)
    return np.interp(sample_times, times, signal)


def apply_to_all(function, signal_array, *args, **kwargs):
    return np.stack([function(signal_array[:, i], *args, **kwargs)
                     for i in range(signal_array.shape[1])], 1)


def filter_with_reflect_pad(x, fs, mains_freq, pad_seconds=1.0):
    """Gaddy avoids filtfilt edge effects by concatenating the neighbouring
    utterances, but EMG-VCTK utterance numbers are not guaranteed to be
    contiguous, so reflect padding is used instead."""
    pad = min(int(pad_seconds * fs), max(x.shape[0] - 1, 0))
    if pad > 0:
        x = np.pad(x, ((pad, pad), (0, 0)), mode='reflect')
    x = apply_to_all(notch_harmonics, x, mains_freq, fs)
    x = apply_to_all(remove_drift, x, fs)
    if pad > 0:
        x = x[pad:x.shape[0] - pad, :]
    return x


def detect_mains_freq(emg_path, fs=None):
    """Check whether 50 Hz or 60 Hz dominates in the PSD of a baseline (resting)
    recording."""
    fs = fs or FLAGS.vctk_emg_sampling_rate
    x = np.load(emg_path).astype(np.float64)
    if x.ndim == 2 and x.shape[1] < x.shape[0]:
        x = x.mean(axis=1)
    f, pxx = scipy.signal.welch(x, fs=fs, nperseg=min(4096, len(x)))
    out = {}
    for cand in (50, 60):
        band = (f > cand - 2) & (f < cand + 2)
        ref = (f > cand - 12) & (f < cand + 12) & ~band
        out[cand] = float(pxx[band].max() / (np.median(pxx[ref]) + 1e-12))
    return out


# ===========================================================================
# File I/O
# ===========================================================================

def load_emg_array(path, num_channels=None):
    """Return the array as (time, channel) float64."""
    num_channels = num_channels or FLAGS.vctk_num_channels
    a = np.asarray(np.load(path), dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[0] == num_channels and a.shape[1] != num_channels:
        a = a.T
    return a


def load_audio_any(filename, max_frames=None, noisereduce=False):
    """Arbitrary SR (EMG-VCTK is 44.1 kHz) -> 22050 Hz -> 80-bin log-mel (time, 80)."""
    audio, sr = sf.read(filename)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float64)

    if noisereduce:
        try:
            import noisereduce as nr
            audio = nr.reduce_noise(y=audio, sr=sr, stationary=False)
        except ImportError:
            logging.warning('noisereduce is not installed - skipping noise reduction')

    if sr != 22050:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
    audio = np.clip(audio, -1, 1)

    spec = mel_spectrogram(torch.tensor(audio, dtype=torch.float32).unsqueeze(0),
                           1024, 80, 22050, 256, 1024, 0, 8000, center=False)
    mspec = spec.squeeze(0).T.numpy()
    if max_frames is not None and mspec.shape[0] > max_frames:
        mspec = mspec[:max_frames, :]
    return mspec


def _phones_tier(tg):
    if 'phones' in tg:
        return tg['phones']
    for name in tg:
        if 'phone' in name.lower():
            return tg[name]
    raise KeyError('The TextGrid has no phone tier')


def read_phonemes_tolerant(textgrid_fname, max_len):
    """Same as data_utils.read_phonemes, but pads with sil when the length does
    not line up."""
    from textgrids import TextGrid
    sil = phoneme_inventory.index('sil')
    tier = _phones_tier(TextGrid(textgrid_fname))
    n = int(tier[-1].xmax * FRAME_RATE) + 1
    phone_ids = np.full(n, sil, dtype=np.int64)
    for interval in tier:
        phone = interval.text.lower()
        if phone in ('', 'sp', 'spn', 'sil'):
            phone = 'sil'
        if phone and phone[-1] in string.digits:
            phone = phone[:-1]
        if phone not in phoneme_inventory:
            phone = 'sil'
        ph_id = phoneme_inventory.index(phone)
        phone_ids[int(interval.xmin * FRAME_RATE):int(interval.xmax * FRAME_RATE)] = ph_id

    if phone_ids.shape[0] >= max_len:
        return phone_ids[:max_len]
    return np.concatenate([phone_ids,
                           np.full(max_len - phone_ids.shape[0], sil, dtype=np.int64)])


# ===========================================================================
# Load a single utterance (same role as Gaddy's load_utterance)
# ===========================================================================

def load_utterance_vctk(record, limit_length=False):
    fs = record.get('emg_sr') or FLAGS.vctk_emg_sampling_rate

    raw = load_emg_array(record['emg_path'])
    x = filter_with_reflect_pad(raw, fs, FLAGS.vctk_mains_freq)

    emg_orig = apply_to_all(subsample, x, EMG_RAW_SR, fs)      # 689.06 Hz
    emg = apply_to_all(subsample, x, EMG_FEATURE_SR, fs)       # 516.80 Hz

    for c in FLAGS.vctk_remove_channels:
        emg[:, int(c)] = 0
        emg_orig[:, int(c)] = 0

    emg_features = get_emg_features(emg)

    max_frames = min(emg_features.shape[0], 800 if limit_length else float('inf'))
    if record.get('audio_path'):
        # The wav for a silent utterance is almost silence, but it is still used for
        # length alignment (same as Gaddy).
        mfccs = load_audio_any(record['audio_path'], max_frames=max_frames,
                               noisereduce=FLAGS.vctk_noisereduce)
    else:
        mfccs = np.zeros((int(min(emg_features.shape[0], max_frames)), 80), dtype=np.float32)

    if emg_features.shape[0] > mfccs.shape[0]:
        emg_features = emg_features[:mfccs.shape[0], :]
    n = emg_features.shape[0]

    emg = emg[FEATURE_OFFSET:FEATURE_OFFSET + 6 * n, :]
    emg_orig = emg_orig[RAW_OFFSET:RAW_OFFSET + 8 * n, :]
    if emg.shape[0] < 6 * n:
        emg = np.pad(emg, ((0, 6 * n - emg.shape[0]), (0, 0)))
    if emg_orig.shape[0] < 8 * n:
        emg_orig = np.pad(emg_orig, ((0, 8 * n - emg_orig.shape[0]), (0, 0)))
    assert emg_orig.shape[0] == 8 * n

    tg = record.get('textgrid_path')
    if tg and os.path.exists(tg):
        phonemes = read_phonemes_tolerant(tg, n)
    else:
        phonemes = np.zeros(n, dtype=np.int64) + phoneme_inventory.index('sil')

    text = preprocess_text(record.get('text', ''))

    return (mfccs.astype(np.float32), emg_features, text,
            phonemes, emg_orig.astype(np.float32))


# ===========================================================================
# Sampler
# ===========================================================================

class SizeAwareSampler(torch.utils.data.Sampler):
    """Build batches so that the total frame length stays within max_len
    (same role as in read_emg.py).

    The paper's Section 3.3 setup uses a maximum of 8000 frames per batch.
    """

    def __init__(self, emg_dataset, max_len):
        self.dataset = emg_dataset
        self.max_len = max_len

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        batch, batch_length = [], 0
        for idx in indices:
            rec = self.dataset.records[idx]
            # The "skip if there are no ascii characters" rule was removed. Empty
            # transcripts are replaced by a placeholder in preprocess_text, so they
            # stay in the training set.
            length = self.dataset.approx_length(idx)
            if length > self.max_len:
                logging.warning('example %s cannot fit within desired batch length', rec['uid'])
            if length + batch_length > self.max_len:
                if batch:
                    yield batch
                batch, batch_length = [], 0
            batch.append(idx)
            batch_length += length
        # The last incomplete batch is dropped (same as Gaddy)


# ===========================================================================
# Dataset
# ===========================================================================

class EMGVCTKDataset(torch.utils.data.Dataset):
    def __init__(self, limit_length=False, dev=False, test=False,
                 no_testset=False, no_normalizers=False):
        with open(FLAGS.manifest_file, encoding='utf-8') as f:
            all_records = json.load(f)['records']

        want = 'test' if test else ('dev' if dev else 'train')
        # Training mixes in audible utterances to obtain a direct CE signal, while
        # evaluation stays silent-only to match the paper. Tying these two to the
        # same flag would change the evaluation difficulty and break comparability
        # across runs.
        silent_only = (FLAGS.vctk_eval_silent_only if want in ('dev', 'test')
                       else FLAGS.vctk_silent_only)

        # Parallel audible index: (session, prompt_key) -> audible record
        self.voiced_by_session_prompt = {}
        self.voiced_by_prompt = {}
        for r in all_records:
            if r['mode'] == 'audible' and r.get('audio_path'):
                self.voiced_by_session_prompt.setdefault((r['session'], r['prompt_key']), r)
                self.voiced_by_prompt.setdefault(r['prompt_key'], r)

        records, n_orphan = [], 0
        for r in all_records:
            if r['mode'] == 'baseline':
                continue
            if not no_testset and r['split'] != want:
                continue
            if silent_only and r['mode'] != 'silent':
                continue
            if r['mode'] == 'silent' and self._find_voiced(r) is None:
                n_orphan += 1
                continue
            if r['mode'] == 'audible' and not r.get('audio_path'):
                continue
            records.append(r)
        if n_orphan:
            logging.warning('[%s] silent utterances excluded for lack of a parallel audible: %d',
                            want, n_orphan)

        records.sort(key=lambda x: x['uid'])
        random.seed(0)
        random.shuffle(records)
        self.records = records
        if not records:
            raise RuntimeError(f'Zero utterances selected (split={want}). Check the manifest.')
        n_sil = sum(1 for r in records if r['mode'] == 'silent')
        logging.info('[%s] %d utterances (silent %d / audible %d)%s',
                     want, len(records), n_sil, len(records) - n_sil,
                     '  <- silent only' if silent_only else '')

        self.no_normalizers = no_normalizers
        self.raw_scale = None
        if not self.no_normalizers:
            with open(FLAGS.normalizers_file, 'rb') as f:
                loaded = pickle.load(f)
            if len(loaded) == 3:
                self.mfcc_norm, self.emg_norm, self.raw_scale = loaded
            else:
                self.mfcc_norm, self.emg_norm = loaded

        sample_mfccs, sample_emg, _, _, _ = load_utterance_vctk(records[0])
        self.num_speech_features = sample_mfccs.shape[1]
        self.num_features = sample_emg.shape[1]
        self.limit_length = limit_length
        self.num_sessions = len({r['session'] for r in all_records})
        self.text_transform = TextTransform()
        self._length_cache = {}
        # For log-output compatibility with transduction_model.py
        self.example_indices = [(r['session_name'], r['uid']) for r in records]

    # ---- Parallel pairs --------------------------------------------------
    def _find_voiced(self, record):
        return (self.voiced_by_session_prompt.get((record['session'], record['prompt_key']))
                or self.voiced_by_prompt.get(record['prompt_key']))

    # ---- Raw EMG scale ---------------------------------------------------
    def _scale_raw(self, raw_emg):
        if FLAGS.vctk_raw_emg_divisor > 0:
            div = FLAGS.vctk_raw_emg_divisor
        elif self.raw_scale:
            div = self.raw_scale / 10.0     # so the top 0.5% of amplitudes land around 10
        else:
            div = 20.0                       # Gaddy default
        raw_emg = raw_emg / div
        return 50 * np.tanh(raw_emg / 50.)   # squashes only the extreme outliers

    # ---- Length estimate (for SizeAwareSampler) --------------------------
    def approx_length(self, idx):
        if idx in self._length_cache:
            return self._length_cache[idx]
        rec = self.records[idx]
        try:
            n_samples = np.load(rec['emg_path'], mmap_mode='r').shape[0]
        except Exception:  # noqa: BLE001
            n_samples = (rec.get('emg_sr') or 800) * 4
        n = int(n_samples * FRAME_RATE / (rec.get('emg_sr') or 800))
        self._length_cache[idx] = n
        return n

    # ---- Standard Dataset API --------------------------------------------
    def silent_subset(self):
        result = copy.copy(self)
        result.records = [r for r in self.records if r['mode'] == 'silent']
        return result

    def subset(self, fraction):
        result = copy.copy(self)
        result.records = self.records[:int(fraction * len(self.records))]
        return result

    def __len__(self):
        return len(self.records)

    @lru_cache(maxsize=None)
    def __getitem__(self, i):
        rec = self.records[i]
        silent = rec['mode'] == 'silent'

        # text comes out of load_utterance_vctk already passed through preprocess_text.
        mfccs, emg, text, phonemes, raw_emg = load_utterance_vctk(rec, self.limit_length)

        # In no_normalizers mode the raw signal is left untouched. make_normalizers
        # estimates the scale here, and measuring an already-scaled value would be
        # self-referential.
        if not self.no_normalizers:
            raw_emg = self._scale_raw(raw_emg)
            mfccs = self.mfcc_norm.normalize(mfccs)
            emg = self.emg_norm.normalize(emg)
            emg = 8 * np.tanh(emg / 8.)

        session_ids = np.full(emg.shape[0], rec['session'], dtype=np.int64)
        # The placeholder '.' is stripped by RemovePunctuation inside TextTransform,
        # so text_int ends up as an empty array (see Note 1 at the top of this file).
        text_int = np.array(self.text_transform.text_to_int(text), dtype=np.int64)

        result = {
            'audio_features': torch.from_numpy(mfccs),
            'emg': torch.from_numpy(emg),
            'text': text,
            'text_int': torch.from_numpy(text_int),
            'file_label': rec['uid'],
            'session_ids': torch.from_numpy(session_ids),
            'book_location': (rec['session'], rec['prompt_key']),
            'silent': silent,
            'raw_emg': torch.from_numpy(raw_emg),
        }
        audio_file = rec.get('audio_path')

        if silent:
            # The target mel and phoneme labels come from the audible recording of
            # the same sentence.
            voiced = self._find_voiced(rec)
            v_mfccs, v_emg, _, phonemes, _ = load_utterance_vctk(voiced, False)
            if not self.no_normalizers:
                v_mfccs = self.mfcc_norm.normalize(v_mfccs)
                v_emg = self.emg_norm.normalize(v_emg)
                v_emg = 8 * np.tanh(v_emg / 8.)
            result['parallel_voiced_audio_features'] = torch.from_numpy(v_mfccs)
            result['parallel_voiced_emg'] = torch.from_numpy(v_emg)
            audio_file = voiced.get('audio_path')

        result['phonemes'] = torch.from_numpy(phonemes)
        result['audio_file'] = audio_file
        return result

    @staticmethod
    def collate_raw(batch):
        audio_features, audio_feature_lengths, parallel_emg = [], [], []
        for ex in batch:
            if ex['silent']:
                audio_features.append(ex['parallel_voiced_audio_features'])
                audio_feature_lengths.append(ex['parallel_voiced_audio_features'].shape[0])
                parallel_emg.append(ex['parallel_voiced_emg'])
            else:
                audio_features.append(ex['audio_features'])
                audio_feature_lengths.append(ex['audio_features'].shape[0])
                parallel_emg.append(np.zeros(1))
        return {
            'audio_features': audio_features,
            'audio_feature_lengths': audio_feature_lengths,
            'emg': [ex['emg'] for ex in batch],
            'raw_emg': [ex['raw_emg'] for ex in batch],
            'parallel_voiced_emg': parallel_emg,
            'phonemes': [ex['phonemes'] for ex in batch],
            'session_ids': [ex['session_ids'] for ex in batch],
            'lengths': [ex['emg'].shape[0] for ex in batch],
            'silent': [ex['silent'] for ex in batch],
            'text': [ex['text'] for ex in batch],
            'text_int': [ex['text_int'] for ex in batch],
            'text_int_lengths': [ex['text_int'].shape[0] for ex in batch],
        }


def make_normalizers(n_samples=200):
    """Estimate the mel/EMG normalizers and the robust raw EMG scale from the
    train split."""
    dataset = EMGVCTKDataset(no_normalizers=True)
    mfcc_samples, emg_samples, raw_abs = [], [], []
    for i in range(min(n_samples, len(dataset))):
        d = dataset[i]
        mfcc_samples.append(d['audio_features'])
        emg_samples.append(d['emg'])
        raw_abs.append(np.abs(d['raw_emg'].numpy()).ravel()[::13])   # subsampled
    mfcc_norm = FeatureNormalizer(mfcc_samples, share_scale=True)
    emg_norm = FeatureNormalizer(emg_samples, share_scale=False)
    raw_scale = float(np.percentile(np.concatenate(raw_abs), 99.5))
    with open(FLAGS.normalizers_file, 'wb') as f:
        pickle.dump((mfcc_norm, emg_norm, raw_scale), f)
    print(f'[OK] normalizers -> {FLAGS.normalizers_file} '
          f'(raw EMG p99.5 = {raw_scale:.3f}, divisor {raw_scale/10:.4f})')


if __name__ == '__main__':
    FLAGS(sys.argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if not os.path.exists(FLAGS.normalizers_file):
        make_normalizers()
    d = EMGVCTKDataset()
    print(f'{len(d)} utterances, {d.num_features} EMG feats, '
          f'{d.num_speech_features} mel bins, {d.num_sessions} sessions')
    for i in range(min(5, len(d))):
        ex = d[i]
        print(f"  {ex['file_label']:38s} emg={tuple(ex['emg'].shape)} "
              f"raw={tuple(ex['raw_emg'].shape)} mel={tuple(ex['audio_features'].shape)} "
              f"ph={tuple(ex['phonemes'].shape)} silent={ex['silent']} "
              f"'{ex['text'][:34]}'")