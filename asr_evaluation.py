import os
import logging
import re

import jiwer
import torch
import tqdm
from unidecode import unidecode

from mel_pipeline import predict_mel_single

from absl import flags
FLAGS = flags.FLAGS

removeBlanks = True


def preprocess_text(text):
    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])

    text = unidecode(text)

    text = text.replace('-', ' ')
    text = text.replace(':', ' ')

    text = transformation(text)

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def compute_wer_cer(targets, predictions, output_directory=None):
    targets = [preprocess_text(t) for t in targets]
    predictions = [preprocess_text(p) for p in predictions]

    if removeBlanks:
        filtered_targets = []
        filtered_predictions = []
        for targ, pred in zip(targets, predictions):
            if targ != '':
                filtered_targets.append(targ)
                filtered_predictions.append(pred)
            else:
                logging.warning(f"Removing empty target and its corresponding prediction: '{pred}'")
        targets = filtered_targets
        predictions = filtered_predictions

    logging.info(f'Sample targets: {targets[:3]}...')
    logging.info(f'Sample predictions: {predictions[:3]}...')

    if len(targets) == 0:
        logging.error('No valid targets found for WER/CER calculation!')
        return None, None

    wer_score = jiwer.wer(targets, predictions)
    cer_score = jiwer.cer(targets, predictions)

    print(f'\n{"=" * 60}')
    print(f'Final WER: {wer_score * 100:.3f}%')
    print(f'Final CER: {cer_score * 100:.3f}%')
    print(f'{"=" * 60}\n')

    logging.info(f'WER: {wer_score}')
    logging.info(f'CER: {cer_score}')

    if output_directory is not None:
        with open(os.path.join(output_directory, 'predictions.txt'), 'w') as f:
            for t, p in zip(targets, predictions):
                f.write(f'REF: {t}\nHYP: {p}\n\n')

    return wer_score, cer_score


def transcribe_mels(asr, mels, batch_size=None, desc='Transcribe'):
    """mels: list of (T, num_speech_features) tensors -> list of str"""
    batch_size = batch_size or FLAGS.asr_batch_size
    predictions = []
    for i in tqdm.tqdm(range(0, len(mels), batch_size), desc, disable=None):
        predictions.extend(asr.transcribe(mels[i:i + batch_size]))
    return predictions


def evaluate(testset, transduction_model, asr, device='cuda',
             output_directory=None, save_mels=False, oracle=False):
    """
    EMG -> transduction model -> mel -> ASR -> text -> WER/CER

    asr    : ParakeetMelASR (the .transcribe interface)
    oracle : if True, feed the ground-truth (parallel audible) mel instead of the
             predicted mel to measure the upper bound of the ASR backend itself.
    """
    mels = []
    targets = []
    desc = 'Oracle mel' if oracle else 'Predict mel'
    for i, datapoint in enumerate(tqdm.tqdm(testset, desc, disable=None)):
        if oracle:
            key = ('parallel_voiced_audio_features'
                   if 'parallel_voiced_audio_features' in datapoint else 'audio_features')
            mel = datapoint[key]
        else:
            mel = predict_mel_single(transduction_model, datapoint, device)
        mels.append(mel.detach().cpu())
        targets.append(str(datapoint['text']))

        if save_mels and output_directory is not None:
            torch.save(mel.detach().cpu(),
                       os.path.join(output_directory, f'example_output_{i}_mel.pt'))

    logging.info('Transcribing %d utterances...', len(mels))
    predictions = transcribe_mels(asr, mels)

    return compute_wer_cer(targets, predictions, output_directory)


def evaluate_from_texts(targets, predictions, output_directory=None):
    """For when you only want to score already-extracted texts."""
    return compute_wer_cer(targets, predictions, output_directory)