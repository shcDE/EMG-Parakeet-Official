import os
import re
import sys
import json
import string
import hashlib
import logging
from collections import defaultdict, Counter

from absl import flags

FLAGS = flags.FLAGS


def _define_string(name, default, helptext):
    """Avoid DuplicateFlagError when imported together with read_emg_vctk.py."""
    if name not in FLAGS:
        flags.DEFINE_string(name, default, helptext)


_define_string(
    'emg_vctk_root',
    '/home/shcheong24/Desktop/experiment_shc/first_author/speech_synthesis_emg/'
    'jbsu2-osfstorage-archive/EMG-VCTK',
    'EMG-VCTK dataset root')
_define_string('manifest_file', 'emg_vctk_manifest.json', 'output manifest path')
flags.DEFINE_bool('vctk_include_baseline', False,
                  'Include the baseline (30 s resting) recordings in the manifest')
flags.DEFINE_bool('vctk_enforce_prompt_disjoint', True,
                  'Move sentences that span several splits into the stricter split '
                  '(test > dev > train) to block leakage')

UID_RE = re.compile(
    r'^(?P<spk>SPK\d+)_(?P<corpus>.+?)-SESS(?P<sess>\d+)_(?P<utt>.+?)_'
    r'(?P<mode>audible|silent|baseline)$')

SPLIT_DIRS = {'train': 'train', 'valid': 'dev', 'test': 'test', 'baseline': 'baseline'}
PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def normalize_text(text):
    return ' '.join(text.lower().translate(PUNCT_TABLE).split())


def prompt_key(text_norm):
    return 'tx_' + hashlib.md5(text_norm.encode('utf-8')).hexdigest()[:12]


def _read_text(path):
    if not os.path.exists(path):
        return ''
    with open(path, encoding='utf-8', errors='replace') as f:
        return ' '.join(f.read().split())


def _find_textgrid(base):
    for ext in ('.TextGrid', '.textgrid', '.Textgrid'):
        if os.path.exists(base + ext):
            return base + ext
    return None


def build_manifest(root):
    records = []
    n_missing = Counter()

    for session_dir in sorted(os.listdir(root)):
        session_path = os.path.join(root, session_dir)
        if not os.path.isdir(session_path):
            continue
        for split_dir in sorted(os.listdir(session_path)):
            split = SPLIT_DIRS.get(split_dir)
            if split is None:
                logging.warning('Skipping unrecognized subdirectory: %s/%s',
                                session_dir, split_dir)
                continue
            if split == 'baseline' and not FLAGS.vctk_include_baseline:
                continue

            d = os.path.join(session_path, split_dir)
            for fn in sorted(os.listdir(d)):
                if not fn.endswith('.json'):
                    continue
                uid = fn[:-len('.json')]
                m = UID_RE.match(uid)
                if m is None:
                    logging.warning('Filename does not match the convention: %s', uid)
                    n_missing['bad_name'] += 1
                    continue

                base = os.path.join(d, uid)
                meta = {}
                try:
                    with open(base + '.json', encoding='utf-8') as f:
                        meta = json.load(f)
                except Exception as e:  # noqa: BLE001
                    logging.warning('Failed to parse json %s (%r)', uid, e)

                emg_path = base + '_emg.npy'
                wav_path = base + '.wav'
                if not os.path.exists(emg_path):
                    n_missing['emg'] += 1
                    continue
                if not os.path.exists(wav_path):
                    n_missing['wav'] += 1
                    wav_path = None

                text = meta.get('text') or meta.get('transcript') or _read_text(base + '.txt')
                if not text:
                    n_missing['text'] += 1
                tnorm = normalize_text(text)

                records.append({
                    'uid': uid,
                    'speaker': m.group('spk'),
                    'session': int(m.group('sess')) - 1,      # 0-based
                    'session_name': session_dir,
                    'mode': meta.get('speaking_mode') or m.group('mode'),
                    'utt_id': m.group('utt'),
                    'split': split,
                    'text': text,
                    'text_norm': tnorm,
                    'prompt_key': prompt_key(tnorm) if tnorm else 'utt_' + uid,
                    'emg_path': emg_path,
                    'emg_sr': int(meta.get('emg_sr', 800)),
                    'audio_path': wav_path,
                    'textgrid_path': _find_textgrid(base),
                })

    if not records:
        raise RuntimeError(f'Found no utterances at all: {root}')
    if n_missing:
        logging.warning('Missing-item tally: %s', dict(n_missing))

    # Remap session indices densely to 0..N-1 (used as session embedding indices)
    remap = {s: i for i, s in enumerate(sorted({r['session'] for r in records}))}
    for r in records:
        r['session'] = remap[r['session']]

    _resolve_prompt_leakage(records)
    _check_pairing(records)
    return records


def _resolve_prompt_leakage(records):
    """Unify a sentence that spans several splits into the stricter split.

    According to the paper the test sentences are shared across all sessions, and
    the sentences of the first ten sessions recur in the later ten. This checks
    whether the released directory split handled that recurrence consistently, and
    moves only the sentences that disagree.
    """
    order = {'test': 3, 'dev': 2, 'train': 1, 'baseline': 0}
    by_prompt = defaultdict(set)
    for r in records:
        by_prompt[r['prompt_key']].add(r['split'])

    crossing = {k: v for k, v in by_prompt.items() if len(v - {'baseline'}) > 1}
    if not crossing:
        logging.info('No sentence-level split leakage (the directory split is consistent)')
        return
    logging.warning('Found %d sentences spanning several splits', len(crossing))
    if not FLAGS.vctk_enforce_prompt_disjoint:
        logging.warning('--vctk_enforce_prompt_disjoint=False, so leaving them as they are')
        return
    target = {k: max(v - {'baseline'}, key=lambda s: order[s]) for k, v in crossing.items()}
    moved = 0
    for r in records:
        t = target.get(r['prompt_key'])
        if t and r['split'] not in (t, 'baseline'):
            r['split'] = t
            moved += 1
    logging.warning('Moved %d utterances into the stricter split', moved)


def _check_pairing(records):
    """Verify every silent utterance has an audible partner in the same session
    with the same sentence."""
    audible = {}
    for r in records:
        if r['mode'] == 'audible':
            audible.setdefault((r['session'], r['prompt_key']), r)

    orphan, no_tg = 0, 0
    for r in records:
        if r['mode'] != 'silent':
            continue
        partner = audible.get((r['session'], r['prompt_key']))
        if partner is None:
            orphan += 1
        elif not partner['textgrid_path']:
            no_tg += 1
    if orphan:
        logging.warning('Silent utterances with no parallel audible: %d '
                        '(these are excluded from training)', orphan)
    if no_tg:
        logging.warning('Silent utterances whose partner audible has no TextGrid: %d '
                        '(the phoneme aux loss is filled with sil)', no_tg)

    # If two different sentences normalize to the same text, parallel pairs could
    # be matched incorrectly.
    per = Counter((r['session'], r['prompt_key']) for r in records if r['mode'] == 'audible')
    dup = [k for k, c in per.items() if c > 1]
    if dup:
        logging.warning('Sessions containing two or more audible utterances with the same '
                        'text: %d (the first one is used as the target)', len(dup))


def summarize(records):
    counts = Counter((r['split'], r['mode']) for r in records)
    print('=' * 64)
    print(f'{"split":10s} {"mode":10s} {"utterances":>12s}')
    print('-' * 64)
    for k in sorted(counts):
        print(f'{k[0]:10s} {k[1]:10s} {counts[k]:12d}')
    print('-' * 64)
    print(f'{"TOTAL":21s} {len(records):12d}')
    print(f'{len({r["session"] for r in records})} sessions / '
          f'{len({r["prompt_key"] for r in records})} sentences / '
          f'{sum(1 for r in records if r["textgrid_path"])} TextGrids')
    print(f'EMG sampling rate: {dict(Counter(r["emg_sr"] for r in records))}')
    print('-' * 64)
    print('Reference values from Table 1 of the paper -> silent 5830/320/476, '
          'audible 2919/160/480 (train/valid/test)')
    print('=' * 64)


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    records = build_manifest(FLAGS.emg_vctk_root)
    summarize(records)
    with open(FLAGS.manifest_file, 'w', encoding='utf-8') as f:
        json.dump({'root': FLAGS.emg_vctk_root, 'records': records},
                  f, indent=1, ensure_ascii=False)
    print(f'[OK] manifest -> {FLAGS.manifest_file}')


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()