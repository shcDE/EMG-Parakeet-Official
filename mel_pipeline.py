import torch

from align import align_from_distances
from data_utils import decollate_tensor, combine_fixed_length

from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('align_output', False,
                     'DTW-align the predicted mel of a silent utterance to the timing of '
                     'the parallel voiced mel. This consumes ground-truth audio at '
                     'inference time and is NOT the Gaddy et al. protocol, which uses only '
                     'silent EMG at test time -- see the module docstring. Keep False '
                     'unless you intend to report the dependency. When False, the mel keeps '
                     'its own EMG-derived length.')

SEQ_LEN = 200


def _target_features(batch, idx):
    """The parallel voiced mel that a silent sample is aligned against."""
    if 'parallel_voiced_audio_features' in batch:
        return batch['parallel_voiced_audio_features'][idx]
    return batch['audio_features'][idx]


def forward_batch(model, batch, device, seq_len=SEQ_LEN):
    """Transduction model forward pass. Returns (combined pred, phoneme_pred)."""
    X = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['emg']], seq_len)
    X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len * 8)
    sess = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len)
    return model(X, X_raw, sess)


def predict_mel_batch(model, batch, device, align=None, seq_len=SEQ_LEN):
    """
    return: (mels, pred, phoneme_pred)
      mels          : list of (T_i, num_speech_features) tensors, in the stage-1
                      normalized mel space
      pred          : raw prediction in combine_fixed_length form (for the
                      alignment loss)
      phoneme_pred  : auxiliary phoneme logits
    """
    align = FLAGS.align_output if align is None else align

    pred, phoneme_pred = forward_batch(model, batch, device, seq_len)
    preds = decollate_tensor(pred, batch['lengths'])

    mels = []
    for i, (p, silent) in enumerate(zip(preds, batch['silent'])):
        if silent and align:
            y = _target_features(batch, i).to(device)
            costs = torch.cdist(p.unsqueeze(0), y.unsqueeze(0)).squeeze(0)
            alignment = align_from_distances(costs.T.detach().cpu().numpy())
            p = p[alignment]
        mels.append(p)
    return mels, pred, phoneme_pred


@torch.no_grad()
def predict_mel_single(model, datapoint, device, align=None):
    """For a single datapoint such as devset[i]."""
    align = FLAGS.align_output if align is None else align

    was_training = model.training
    model.eval()

    silent = datapoint['silent']
    sess = datapoint['session_ids'].to(device).unsqueeze(0)
    X = datapoint['emg'].to(device).unsqueeze(0)
    X_raw = datapoint['raw_emg'].to(device).unsqueeze(0)

    pred, _ = model(X, X_raw, sess)
    pred = pred.squeeze(0)

    if silent and align:
        key = ('parallel_voiced_audio_features'
               if 'parallel_voiced_audio_features' in datapoint else 'audio_features')
        y = datapoint[key].to(device).unsqueeze(0)
        costs = torch.cdist(pred.unsqueeze(0), y).squeeze(0)
        alignment = align_from_distances(costs.T.detach().cpu().numpy())
        pred = pred[alignment]

    if was_training:
        model.train()
    return pred