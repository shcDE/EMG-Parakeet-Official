import sys
import json
import logging
from collections import Counter

import numpy as np
import torch
import tqdm

from absl import flags

from data_utils import phoneme_inventory, combine_fixed_length, decollate_tensor
from read_emg_vctk import EMGVCTKDataset

FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint', None,
                    'Path to a trained model.pt (statistics only if omitted)')
flags.DEFINE_integer('diag_n', 320, 'Number of dev utterances to inspect')

# This flag is normally defined by transduction_model.py, so it is defined here
# directly. It must match the value used during training for the DTW alignment to
# be reproduced identically.
if 'phoneme_loss_weight' not in FLAGS:
    flags.DEFINE_float('phoneme_loss_weight', 0.5,
                       'Weight of the phoneme term in the DTW cost (match the value '
                       'used during training)')

SIL = phoneme_inventory.index('sil')


def coverage():
    print('=' * 70)
    print('1. TextGrid coverage')
    print('=' * 70)
    with open(FLAGS.manifest_file, encoding='utf-8') as f:
        recs = json.load(f)['records']

    audible_tg = {}
    for r in recs:
        if r['mode'] == 'audible':
            audible_tg[(r['session'], r['prompt_key'])] = bool(r['textgrid_path'])

    per_split = {}
    for r in recs:
        if r['mode'] != 'silent':
            continue
        has = audible_tg.get((r['session'], r['prompt_key']))
        d = per_split.setdefault(r['split'], Counter())
        d['total'] += 1
        d['with_tg'] += 1 if has else 0

    for split, c in sorted(per_split.items()):
        pct = 100 * c['with_tg'] / max(c['total'], 1)
        mark = 'OK' if pct > 99 else 'WARN'
        print(f'  [{mark}] {split:6s} {c["with_tg"]:5d} of {c["total"]:5d} silent '
              f'utterances have a TextGrid ({pct:.1f}%)')


def target_distribution(dataset):
    print('\n' + '=' * 70)
    print('2. Phoneme distribution of the dev targets')
    print('=' * 70)
    counts = np.zeros(len(phoneme_inventory), dtype=np.int64)
    n_all_sil = 0
    for i in tqdm.tqdm(range(min(FLAGS.diag_n, len(dataset))), 'target scan', disable=None):
        ph = dataset[i]['phonemes'].numpy()
        counts += np.bincount(ph, minlength=len(phoneme_inventory))
        if (ph == SIL).all():
            n_all_sil += 1

    total = counts.sum()
    sil_frac = counts[SIL] / total
    print(f'  {total:,} frames total, {int((counts > 0).sum())} distinct phonemes')
    print(f'  Utterances that are entirely sil: {n_all_sil}'
          + ('  <- suspect a TextGrid parsing failure!' if n_all_sil else ''))
    print(f'\n  >> sil fraction = {sil_frac*100:.2f}%  '
          f'(= the accuracy of a model that always predicts sil)')
    print('\n  Top 10 phonemes:')
    for idx in np.argsort(-counts)[:10]:
        print(f'    {phoneme_inventory[idx]:4s} {counts[idx]:9,d}  {counts[idx]/total*100:5.2f}%')
    return sil_frac


def model_behavior(dataset, sil_frac):
    if not FLAGS.checkpoint:
        print('\n(pass --checkpoint to also run the prediction analysis)')
        return
    print('\n' + '=' * 70)
    print('3-5. Prediction analysis')
    print('=' * 70)

    from architecture import Model
    from align import align_from_distances

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(dataset.num_features, dataset.num_speech_features,
                  len(phoneme_inventory)).to(device)
    state = torch.load(FLAGS.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    pred_counts = np.zeros(len(phoneme_inventory), dtype=np.int64)
    confusion = np.zeros((len(phoneme_inventory), len(phoneme_inventory)))
    correct = total = 0
    seq_len = 200

    dl = torch.utils.data.DataLoader(dataset, batch_size=16,
                                     collate_fn=dataset.collate_raw)
    with torch.no_grad():
        for batch in tqdm.tqdm(dl, 'predict', disable=None):
            X = combine_fixed_length([t.to(device) for t in batch['emg']], seq_len)
            Xr = combine_fixed_length([t.to(device) for t in batch['raw_emg']], seq_len * 8)
            sess = combine_fixed_length([t.to(device) for t in batch['session_ids']], seq_len)
            pred, ph_pred = model(X, Xr, sess)
            preds = decollate_tensor(pred, batch['lengths'])
            ph_preds = decollate_tensor(ph_pred, batch['lengths'])

            for p, pp, y, yph in zip(preds, ph_preds, batch['audio_features'],
                                     batch['phonemes']):
                y = y.to(device)
                costs = torch.cdist(p.unsqueeze(0), y.unsqueeze(0)).squeeze(0)
                lp = torch.nn.functional.log_softmax(pp, -1)
                costs = costs + FLAGS.phoneme_loss_weight * -lp[:, yph.to(device)]
                al = align_from_distances(costs.T.cpu().numpy())
                pa = lp.argmax(-1)[al].cpu().numpy()
                ya = yph.numpy()
                pred_counts += np.bincount(pa, minlength=len(phoneme_inventory))
                correct += int((pa == ya).sum())
                total += len(ya)
                for a, b in zip(pa, ya):
                    confusion[a, b] += 1

    acc = correct / total
    print(f'\n  overall accuracy   : {acc*100:.2f}%')
    print(f'  sil-only baseline  : {sil_frac*100:.2f}%')
    gain = (acc - sil_frac) * 100

    # confusion is indexed [predicted, target]. sil dominates the metric, so it is
    # broken out separately.
    tgt_counts = confusion.sum(0)
    nonsil_total = tgt_counts.sum() - tgt_counts[SIL]
    nonsil_correct = sum(confusion[i, i] for i in range(len(phoneme_inventory))
                         if i != SIL)
    nonsil_acc = nonsil_correct / max(nonsil_total, 1)

    # Expected accuracy of a model that matches the marginal distribution but
    # guesses at random = sum(p_i^2)
    p_tgt = tgt_counts / tgt_counts.sum()
    chance = float((p_tgt ** 2).sum())

    recalls = [confusion[i, i] / tgt_counts[i]
               for i in range(len(phoneme_inventory)) if tgt_counts[i] > 0]
    balanced = float(np.mean(recalls))

    sil_recall = confusion[SIL, SIL] / max(tgt_counts[SIL], 1)
    sil_prec = confusion[SIL, SIL] / max(pred_counts[SIL], 1)

    print(f'\n  -- metrics with the sil-dominance effect removed --')
    print(f'  accuracy excl. sil : {nonsil_acc*100:.2f}%  '
          f'(over the {int(nonsil_total):,} non-sil frames)')
    print(f'  balanced accuracy  : {balanced*100:.2f}%  (mean per-class recall)')
    print(f'  marginal-random    : {chance*100:.2f}%  (the zero-information floor)')
    print(f'  sil recall/precision: {sil_recall*100:.1f}% / {sil_prec*100:.1f}%')

    if nonsil_acc < 0.05:
        verdict = ('nothing but sil is being predicted correctly -> the phoneme head is '
                   'effectively untrained')
    elif gain < 3:
        verdict = ('negligible by overall accuracy, but the non-sil performance is '
                   'meaningful -> the metric is misleading here')
    else:
        verdict = (f'+{gain:.1f}pp over the baseline, {nonsil_acc*100:.1f}% excluding sil '
                   f'-> it is learning')
    print(f'\n  >> {verdict}')

    p = pred_counts / pred_counts.sum()
    ent = -(p[p > 0] * np.log(p[p > 0])).sum()
    ent_tgt = -(p_tgt[p_tgt > 0] * np.log(p_tgt[p_tgt > 0])).sum()
    print(f'\n  prediction entropy : {ent:.3f} nats '
          f'(target {ent_tgt:.3f} / uniform {np.log(len(phoneme_inventory)):.3f})')
    print(f'  distinct classes predicted: {int((pred_counts > 0).sum())} / '
          f'{len(phoneme_inventory)}')
    print(f'  most frequent prediction  : {p.max()*100:.1f}% ({phoneme_inventory[p.argmax()]})')
    if p.max() > 0.8:
        print('  >> it has collapsed onto a single class')

    print('\n  Top confusion pairs (predicted <-> target):')
    print('  (pairs differing only in voicing, such as z<->s or f<->v, are in principle '
          'indistinguishable from silent EMG)')
    tc = confusion.sum(0) + 1e-4
    pairs = []
    for a in range(len(phoneme_inventory)):
        for b in range(a):
            v = (confusion[a, b] + confusion[b, a]) / (tc[a] + tc[b])
            pairs.append((v, a, b))
    for v, a, b in sorted(pairs)[-8:]:
        print(f'    {phoneme_inventory[a]:4s} <-> {phoneme_inventory[b]:4s}  {v*100:5.1f}%')


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    coverage()
    dev = EMGVCTKDataset(dev=True)
    sil_frac = target_distribution(dev)
    model_behavior(dev, sil_frac)


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()