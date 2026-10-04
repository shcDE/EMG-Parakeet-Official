import os
import sys
import json
from collections import Counter, defaultdict

import numpy as np

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'emg_vctk_root',
    '/home/shcheong24/Desktop/experiment_shc/first_author/speech_synthesis_emg/'
    'jbsu2-osfstorage-archive/EMG-VCTK',
    'EMG-VCTK dataset root')
flags.DEFINE_string('scan_output', 'emg_vctk_layout.json', 'where to dump the layout report')
flags.DEFINE_integer('tree_depth', 3, 'how deep to print the directory tree')
flags.DEFINE_integer('n_samples', 3, 'how many sample files to inspect per extension')

AUDIO_EXTS = {'.wav', '.flac', '.ogg', '.mp3'}
ARRAY_EXTS = {'.npy', '.npz'}
TEXTY_EXTS = {'.txt', '.lab', '.json', '.csv', '.tsv', '.textgrid'}


def human(n):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f'{n:.1f}{unit}'
        n /= 1024
    return f'{n:.1f}PB'


def print_tree(root, max_depth):
    print('=' * 78)
    print(f'DIRECTORY TREE (depth <= {max_depth})')
    print('=' * 78)
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames.sort()
        indent = '  ' * depth
        name = os.path.basename(dirpath) or dirpath
        exts = Counter(os.path.splitext(f)[1].lower() for f in filenames)
        ext_str = ', '.join(f'{e or "<noext>"}:{c}' for e, c in exts.most_common(6))
        print(f'{indent}{name}/   [{len(filenames)} files] {ext_str}')
        if depth == max_depth and dirnames:
            print(f'{indent}  ... ({len(dirnames)} more subdirs, truncated)')


def collect(root):
    by_ext = defaultdict(list)
    total_bytes = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            p = os.path.join(dirpath, f)
            ext = os.path.splitext(f)[1].lower()
            by_ext[ext].append(p)
            try:
                total_bytes += os.path.getsize(p)
            except OSError:
                pass
    return by_ext, total_bytes


def inspect_array(path):
    try:
        if path.endswith('.npz'):
            with np.load(path) as z:
                return {'type': 'npz', 'keys': list(z.keys()),
                        'shapes': {k: list(z[k].shape) for k in z.keys()}}
        a = np.load(path, mmap_mode='r')
        return {'type': 'npy', 'shape': list(a.shape), 'dtype': str(a.dtype),
                'min': float(np.asarray(a[:1000]).min()),
                'max': float(np.asarray(a[:1000]).max())}
    except Exception as e:  # noqa: BLE001
        return {'error': repr(e)}


def inspect_audio(path):
    try:
        import soundfile as sf
        info = sf.info(path)
        return {'samplerate': info.samplerate, 'channels': info.channels,
                'frames': info.frames, 'duration_s': round(info.duration, 3),
                'subtype': info.subtype}
    except Exception as e:  # noqa: BLE001
        return {'error': repr(e)}


def head_text(path, n=6):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = [next(f).rstrip('\n') for _ in range(n)]
    except StopIteration:
        pass
    except Exception as e:  # noqa: BLE001
        return {'error': repr(e)}
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = [ln.rstrip('\n') for ln, _ in zip(f, range(n))]
        return {'head': lines}
    except Exception as e:  # noqa: BLE001
        return {'error': repr(e)}


def main():
    root = FLAGS.emg_vctk_root
    if not os.path.isdir(root):
        print(f'[FATAL] not a directory: {root}')
        sys.exit(1)

    print_tree(root, FLAGS.tree_depth)

    by_ext, total_bytes = collect(root)
    print()
    print('=' * 78)
    print(f'FILE INVENTORY  (total {human(total_bytes)})')
    print('=' * 78)
    for ext, paths in sorted(by_ext.items(), key=lambda kv: -len(kv[1])):
        print(f'  {ext or "<noext>":12s} {len(paths):7d}')

    report = {'root': root, 'total_bytes': total_bytes,
              'counts': {k: len(v) for k, v in by_ext.items()}, 'samples': {}}

    print()
    print('=' * 78)
    print('SAMPLE FILE INSPECTION')
    print('=' * 78)
    for ext, paths in sorted(by_ext.items()):
        paths = sorted(paths)
        picks = paths[:FLAGS.n_samples]
        for p in picks:
            rel = os.path.relpath(p, root)
            if ext in ARRAY_EXTS:
                info = inspect_array(p)
            elif ext in AUDIO_EXTS:
                info = inspect_audio(p)
            elif ext in TEXTY_EXTS:
                info = head_text(p)
            else:
                info = {'size': human(os.path.getsize(p))}
            report['samples'].setdefault(ext, []).append({'path': rel, 'info': info})
            print(f'  {rel}')
            print(f'      {json.dumps(info, ensure_ascii=False)[:400]}')

    with open(FLAGS.scan_output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print()
    print(f'[OK] layout report -> {FLAGS.scan_output}')
    print('Next step: build the manifest with '
          'python emg_vctk_index.py --emg_vctk_root=...')


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()