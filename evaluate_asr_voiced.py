import sys
import os
import json
import random
import logging

import torch
from torch import nn

from architecture import Model
from transduction_model import test
from asr_evaluation import evaluate
from data_utils import phoneme_inventory, print_confusion

from read_emg import EMGDataset

from read_emg_vctk import EMGVCTKDataset

try:
    import parakeet_mel_asr  # noqa: F401
except ImportError as _e:
    parakeet_mel_asr = None
    print(f'[warn] parakeet backend unavailable: {_e}')

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_enum('dataset', 'gaddy', ['gaddy', 'vctk'], 'which corpus to evaluate')
flags.DEFINE_list('models', [], 'transduction model checkpoint(s)')
flags.DEFINE_boolean('dev', False, 'evaluate dev instead of test')
flags.DEFINE_boolean('save_eval_mels', False, 'Save the predicted mel as .pt')
flags.DEFINE_boolean('oracle', False,
                     'Feed the ground-truth mel to measure the ASR backend upper bound')
flags.DEFINE_boolean('skip_mel_metrics', False, 'Skip the DTW loss / phoneme accuracy computation')

if 'asr_batch_size' not in FLAGS:
    flags.DEFINE_integer('asr_batch_size', 8, 'ASR transcribe batch size')


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, x_raw, sess):
        ys, ps = [], []
        for model in self.models:
            y, p = model(x, x_raw, sess)
            ys.append(y)
            ps.append(p)
        return torch.stack(ys, 0).mean(0), torch.stack(ps, 0).mean(0)


def build_backend(num_mel_features, device):
    assert parakeet_mel_asr is not None, 'Failed to import the parakeet backend.'
    assert FLAGS.parakeet_checkpoint is not None, \
        'Evaluating without --parakeet_checkpoint is meaningless because the mel adapter would be random.'
    return parakeet_mel_asr.build_parakeet_asr(num_mel_features, device=device)


# ---------------------------------------------------------------------------
# Gaddy: load everything with no_testset, then keep only the non-silent examples
# whose location appears in the dev/test list of testset_file.
# ---------------------------------------------------------------------------
def _build_gaddy_voiced(dev):
    assert EMGDataset is not None, 'failed to import read_emg'
    testset = EMGDataset(dev=False, test=False, no_testset=True)

    with open(FLAGS.testset_file) as f:
        target_locations = json.load(f)['dev' if dev else 'test']

    voiced = []
    for example in testset.example_indices:
        directory_info, idx = example
        if directory_info.silent:
            continue
        with open(os.path.join(directory_info.directory, f'{idx}_info.json')) as f:
            info = json.load(f)
        if [info['book'], info['sentence_index']] in target_locations:
            voiced.append(example)

    assert voiced, 'No voiced examples selected. Check testset_file.'
    voiced.sort()
    random.seed(0)
    random.shuffle(voiced)
    testset.example_indices = voiced
    return testset, len(voiced)


# ---------------------------------------------------------------------------
# VCTK: __len__ / __getitem__ read from records, so records is what must be
# filtered. example_indices holds string tuples for logging only, so editing it
# has no effect on what gets evaluated. Also, audible utterances are dropped at
# construction time unless vctk_eval_silent_only is turned off.
# ---------------------------------------------------------------------------
def _build_vctk_voiced(dev):
    assert EMGVCTKDataset is not None, 'failed to import read_emg_vctk'
    assert not FLAGS.vctk_eval_silent_only, \
        'Voiced evaluation requires --novctk_eval_silent_only.'

    testset = EMGVCTKDataset(dev=dev, test=not dev)

    voiced = [r for r in testset.records if r['mode'] == 'audible']
    assert voiced, 'No audible utterances found. Check the split/mode fields in the manifest.'

    voiced.sort(key=lambda x: x['uid'])
    random.seed(0)
    random.shuffle(voiced)
    testset.records = voiced
    testset.example_indices = [(r['session_name'], r['uid']) for r in voiced]
    testset._length_cache = {}
    return testset, len(voiced)


def build_voiced_testset(dev):
    builder = _build_gaddy_voiced if FLAGS.dataset == 'gaddy' else _build_vctk_voiced
    testset, n = builder(dev)
    logging.info(f'[{FLAGS.dataset}] {n} voiced {"dev" if dev else "test"} examples')
    return testset


def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(handlers=[
        logging.FileHandler(os.path.join(FLAGS.output_directory, 'eval_log.txt'), 'w'),
        logging.StreamHandler()
    ], level=logging.INFO, format="%(message)s")
    logging.info(sys.argv)

    dev = FLAGS.dev
    testset = build_voiced_testset(dev)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ensemble = None
    if not FLAGS.oracle:
        models = []
        for fname in FLAGS.models:
            state_dict = torch.load(fname)
            m = Model(testset.num_features, testset.num_speech_features,
                      len(phoneme_inventory)).to(device)
            m.load_state_dict(state_dict)
            models.append(m)
        assert models, 'Specify transduction model checkpoints with --models.'
        ensemble = EnsembleModel(models)

        if not FLAGS.skip_mel_metrics:
            val_loss, phoneme_acc, confusion = test(ensemble, testset, device)
            logging.info(f'DTW Loss: {val_loss:.4f}')
            logging.info(f'Phoneme Accuracy: {phoneme_acc:.4f} ({phoneme_acc*100:.2f}%)')
            print_confusion(confusion)

    asr = build_backend(testset.num_speech_features, device)

    wer, cer = evaluate(testset, ensemble, asr, device=device,
                        output_directory=FLAGS.output_directory,
                        save_mels=FLAGS.save_eval_mels,
                        oracle=FLAGS.oracle)
    tag = 'ORACLE ' if FLAGS.oracle else ''
    logging.info(f'{tag}[{FLAGS.dataset}/voiced] Final WER {wer*100:.3f}% / CER {cer*100:.3f}%')


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()