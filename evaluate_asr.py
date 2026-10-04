import sys
import os
import logging

import torch
from torch import nn

from architecture import Model
from transduction_model import test
from read_emg import EMGDataset
# from read_emg_vctk import EMGVCTKDataset as EMGDataset
from asr_evaluation import evaluate
from data_utils import phoneme_inventory, print_confusion

# The backend modules must be imported up front so that their flags get registered
# (a deferred import would not have them yet at FLAGS(sys.argv) time, raising
# UnrecognizedFlagError)
try:
    import parakeet_mel_asr  # noqa: F401
except ImportError as _e:  # environment without NeMo installed
    parakeet_mel_asr = None
    print(f'[warn] parakeet backend unavailable: {_e}')

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_list('models', [], 'transduction model checkpoint(s)')
flags.DEFINE_boolean('dev', False, 'evaluate dev instead of test')
flags.DEFINE_boolean('save_eval_mels', False, 'Save the predicted mel as .pt')
flags.DEFINE_boolean('oracle', False,
                     'Feed the ground-truth mel to measure the ASR backend upper bound '
                     '(used to determine whether Stage 1 is the bottleneck)')
flags.DEFINE_boolean('skip_mel_metrics', False, 'Skip the DTW loss / phoneme accuracy computation')

# Registered here because train_*.py is not imported (guarded against duplicate definition)
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
        'Evaluating without --parakeet_checkpoint is meaningless because the mel adapter ' \
        'would be random.'
    return parakeet_mel_asr.build_parakeet_asr(num_mel_features, device=device)

def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(handlers=[
        logging.FileHandler(os.path.join(FLAGS.output_directory, 'eval_log.txt'), 'w'),
        logging.StreamHandler()
    ], level=logging.INFO, format="%(message)s")
    logging.info(sys.argv)

    dev = FLAGS.dev
    testset = EMGDataset(dev=dev, test=not dev)
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
    logging.info(f'{tag} Final WER {wer*100:.3f}% / CER {cer*100:.3f}%')


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()