import os
import sys
import math
import logging
import subprocess

import numpy as np
import torch
import tqdm

from read_emg import EMGDataset, SizeAwareSampler
# from read_emg_vctk import EMGVCTKDataset as EMGDataset, SizeAwareSampler

from architecture import Model
from data_utils import phoneme_inventory
from mel_pipeline import predict_mel_batch
from transduction_model import dtw_loss   # also defines the shared flags (output_directory etc.)
from parakeet_mel_asr import ParakeetMelASR
from asr_evaluation import compute_wer_cer

from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('parakeet_epochs', 20, 'stage-2 epochs')
flags.DEFINE_float('mel_adapter_lr', 3e-4, 'MelAdapter learning rate')
flags.DEFINE_float('nemo_adapter_lr', 1e-3,
                   'NeMo encoder adapter learning rate. NeMo recommends 1e-3 '
                   '(a residual adapter starts near zero output, so a large lr is safe)')
flags.DEFINE_float('parakeet_weight_decay', 0.0,
                   'weight decay. The recommended default for NeMo adapters is 0.0')
flags.DEFINE_integer('parakeet_warmup', 200, 'linear warmup steps')
flags.DEFINE_float('parakeet_min_lr_ratio', 0.01,
                   'lr multiplier at the end of the cosine decay. 0 means a constant lr')
flags.DEFINE_integer('grad_accum', 16, 'gradient accumulation steps')
# Set this back to 8000 when returning to EMG-VCTK (see the dataset switch note above)
flags.DEFINE_integer('emg_batch_tokens', 256000,
                     'SizeAwareSampler length budget. '
                     'read_emg_vctk=8000. The length unit differs per dataset.')
flags.DEFINE_integer('max_utts_per_batch', 2, 'Max utterances per batch (0 = unlimited)')
flags.DEFINE_boolean('joint_finetune', False, 'Also train the transduction model')
flags.DEFINE_float('joint_dtw_weight', 1.0, 'Stage-1 alignment loss weight during joint training')
flags.DEFINE_float('transduction_lr', 1e-5, 'transduction model lr during joint training')
flags.DEFINE_integer('eval_every', 1, 'Compute dev WER every N epochs (0 = loss only)')
flags.DEFINE_integer('eval_max_examples', 0, 'Max samples used for dev WER (0 = all)')
flags.DEFINE_integer('asr_batch_size', 8, 'transcribe batch size during dev evaluation')
flags.DEFINE_integer('spec_augment_start_epoch', 0,
                     'Epoch from which SpecAugment is applied. EMG-predicted mel is '
                     'already degraded, so masking early on can block learning.')
flags.DEFINE_integer('mel_warmup_epochs', 0,
                     'Freeze the NeMo adapters and train only the MelAdapter for the first '
                     'N epochs. Prevents the encoder adapters from converging to a random '
                     'mel distribution first.')

# ---- wandb (matches exp_manager.wandb_logger_kwargs in nemo_config.yaml) ----
flags.DEFINE_boolean('create_wandb_logger', True, 'Enable wandb logging')
flags.DEFINE_string('wandb_name', 'EMG-ASR', 'wandb run name')
flags.DEFINE_string('wandb_project', 'EMG-to-Speech', 'wandb project')
flags.DEFINE_string('wandb_entity', None, 'wandb entity')
flags.DEFINE_boolean('wandb_offline', False, 'offline mode')
flags.DEFINE_list('wandb_tags', [], 'wandb tags')
flags.DEFINE_integer('wandb_log_every', 50, 'Log the train loss every N steps')


def init_wandb(extra_config=None):
    """Initialize wandb. Returns None on failure and training continues as usual."""
    if not FLAGS.create_wandb_logger:
        return None
    try:
        import wandb
    except ImportError:
        logging.warning('wandb is not installed - skipping logging (pip install wandb)')
        return None

    config = {
        'mel_adapter_lr': FLAGS.mel_adapter_lr,
        'nemo_adapter_lr': FLAGS.nemo_adapter_lr,
        'weight_decay': FLAGS.parakeet_weight_decay,
        'warmup': FLAGS.parakeet_warmup,
        'min_lr_ratio': FLAGS.parakeet_min_lr_ratio,
        'epochs': FLAGS.parakeet_epochs,
        'grad_accum': FLAGS.grad_accum,
        'emg_batch_tokens': FLAGS.emg_batch_tokens,
        # MelAdapter architecture. kernel and use_tcs must be logged too, or the
        # separable-vs-dense ablation runs are indistinguishable in the wandb UI.
        'adapter_hidden': FLAGS.adapter_hidden,
        'adapter_layers': FLAGS.adapter_layers,
        'adapter_kernel': FLAGS.adapter_kernel,
        'adapter_use_tcs': FLAGS.adapter_use_tcs,
        'adapter_out_norm': FLAGS.adapter_out_norm,
        'adapter_lower_dim': FLAGS.adapter_lower_dim,
        'adapter_upper_dim': FLAGS.adapter_upper_dim,
        'use_spec_augment': FLAGS.use_spec_augment,
        'freeze_bn_stats': FLAGS.freeze_bn_stats,
        'joint_finetune': FLAGS.joint_finetune,
        'pretrained_model': FLAGS.parakeet_model_name,
        'stage1_checkpoint': FLAGS.start_training_from,
    }
    # The stage-1 alignment loss only affects this run when the transduction model
    # is being updated, so it is logged only then to keep the config clean.
    if FLAGS.joint_finetune:
        config.update({
            'joint_dtw_weight': FLAGS.joint_dtw_weight,
            'transduction_lr': FLAGS.transduction_lr,
            'dtw_loss_type': FLAGS.dtw_loss_type,
            'soft_dtw_gamma': FLAGS.soft_dtw_gamma,
            'soft_dtw_band': FLAGS.soft_dtw_band,
        })
        if FLAGS.dtw_loss_type == 'asa':
            config.update({
                'asa_tau': FLAGS.asa_tau,
                'asa_detach_target': FLAGS.asa_detach_target,
            })
    if extra_config:
        config.update(extra_config)

    try:
        run = wandb.init(
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            name=FLAGS.wandb_name,
            tags=list(FLAGS.wandb_tags) or None,
            mode='offline' if FLAGS.wandb_offline else 'online',
            config=config,
            dir=FLAGS.output_directory,
        )
        logging.info('wandb run: %s', getattr(run, 'url', '(offline)'))
        return wandb
    except Exception as e:  # noqa: BLE001
        logging.warning('wandb initialization failed (%s) - continuing without logging.', e)
        return None


def batch_texts(batch, dataset):
    """Extract the list of ground-truth transcripts from a batch."""
    if 'text' in batch:
        return [str(t) for t in batch['text']]

    tt = getattr(dataset, 'text_transform', None)
    if tt is not None and hasattr(tt, 'int_to_text') and 'text_int' in batch:
        logging.warning("collate_raw has no 'text' key - reconstructing from text_int.")
        return [tt.int_to_text(ti.tolist()) for ti in batch['text_int']]

    raise KeyError("The batch has no ground-truth text. Add the line "
                   "\"'text': [ex['text'] for ex in batch],\" to collate_raw in the dataset "
                   "you are using (neither read_emg.py nor read_emg_vctk.py has it by default).")


def split_batch(batch, max_utts):
    if not max_utts:
        return [batch]
    n = len(batch['lengths'])
    if n <= max_utts:
        return [batch]
    return [{k: (v[s:min(s + max_utts, n)] if isinstance(v, list) else v)
             for k, v in batch.items()}
            for s in range(0, n, max_utts)]


def evaluate_dev(transduction_model, parakeet_asr, devset, device, max_examples=0):
    transduction_model.eval()
    parakeet_asr.eval()

    dataloader = torch.utils.data.DataLoader(
        devset, batch_size=FLAGS.asr_batch_size, collate_fn=devset.collate_raw, shuffle=False)

    targets, predictions = [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, 'Dev WER', disable=None):
            mels, _, _ = predict_mel_batch(transduction_model, batch, device)
            predictions.extend(parakeet_asr.transcribe(mels))
            targets.extend(batch_texts(batch, devset))
            if max_examples and len(targets) >= max_examples:
                break

    transduction_model.train()
    parakeet_asr.train()
    wer, cer = compute_wer_cer(targets, predictions, FLAGS.output_directory)
    return wer, cer, targets, predictions


def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(handlers=[
        logging.FileHandler(os.path.join(FLAGS.output_directory, 'parakeet_train_log.txt'), 'w'),
        logging.StreamHandler()
    ], level=logging.INFO, format="%(message)s")
    logging.info(sys.argv)
    logging.info(subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE,
                                universal_newlines=True).stdout)
    logging.info('key flags: mel_lr=%g nemo_lr=%g mel_adapter=%s hidden=%d layers=%d '
                 'kernel=%d nemo_dim=%d/%d out_norm=%s spec_aug=%s',
                 FLAGS.mel_adapter_lr, FLAGS.nemo_adapter_lr,
                 'TCS' if FLAGS.adapter_use_tcs else 'dense',
                 FLAGS.adapter_hidden, FLAGS.adapter_layers, FLAGS.adapter_kernel,
                 FLAGS.adapter_lower_dim, FLAGS.adapter_upper_dim,
                 FLAGS.adapter_out_norm, FLAGS.use_spec_augment)
    if FLAGS.joint_finetune:
        extra = (f' tau={FLAGS.asa_tau} detach={FLAGS.asa_detach_target}'
                 if FLAGS.dtw_loss_type == 'asa' else '')
        logging.info('joint finetune: transduction_lr=%g stage1_weight=%g '
                     'stage1_loss=%s gamma=%g band=%d%s',
                     FLAGS.transduction_lr, FLAGS.joint_dtw_weight,
                     FLAGS.dtw_loss_type, FLAGS.soft_dtw_gamma, FLAGS.soft_dtw_band, extra)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainset = EMGDataset(dev=False, test=False)
    devset = EMGDataset(dev=True)
    logging.info('train / dev split: %d %d', len(trainset), len(devset))

    # ---------------- Stage 1 model ----------------
    transduction_model = Model(devset.num_features, devset.num_speech_features,
                               len(phoneme_inventory)).to(device)
    assert FLAGS.start_training_from is not None, \
        'Specify the stage-1 transduction model (.pt) with --start_training_from.'
    transduction_model.load_state_dict(torch.load(FLAGS.start_training_from), strict=False)

    if FLAGS.joint_finetune:
        transduction_model.train()
        for p in transduction_model.parameters():
            p.requires_grad = True
    else:
        transduction_model.eval()
        for p in transduction_model.parameters():
            p.requires_grad = False

    # ---------------- Parakeet ----------------
    parakeet_asr = ParakeetMelASR(devset.num_speech_features, device=device)
    if FLAGS.parakeet_checkpoint is not None:
        parakeet_asr.load(FLAGS.parakeet_checkpoint)
    parakeet_asr.train()

    # ---------------- optimizer ----------------
    param_groups = [
        {'params': list(parakeet_asr.mel_adapter.parameters()), 'lr': FLAGS.mel_adapter_lr},
        {'params': [p for p in parakeet_asr.asr.parameters() if p.requires_grad],
         'lr': FLAGS.nemo_adapter_lr},
    ]
    if FLAGS.joint_finetune:
        param_groups.append({'params': list(transduction_model.parameters()),
                             'lr': FLAGS.transduction_lr})

    optim = torch.optim.AdamW(param_groups, betas=(0.9, 0.98),
                              weight_decay=FLAGS.parakeet_weight_decay)
    base_lrs = [g['lr'] for g in optim.param_groups]

    sampler = SizeAwareSampler(trainset, FLAGS.emg_batch_tokens)
    steps_per_epoch = sum(1 for _ in sampler)
    total_steps = max(1, steps_per_epoch * FLAGS.parakeet_epochs)
    logging.info('steps/epoch ~= %d, total steps ~= %d, warmup %d',
                 steps_per_epoch, total_steps, FLAGS.parakeet_warmup)
    if steps_per_epoch > 0.5 * len(trainset):
        logging.warning(
            'steps/epoch (%d) exceeds half the utterance count (%d). --emg_batch_tokens (%d) '
            'may be too small for this dataset\'s length unit. '
            'Gaddy read_emg.py counts 1000 Hz raw samples, so 256000 is appropriate; '
            'read_emg_vctk.py counts 86 Hz mel frames, so 8000 is appropriate.',
            steps_per_epoch, len(trainset), FLAGS.emg_batch_tokens)

    wandb = init_wandb({
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'train_utterances': len(trainset),
        'dev_utterances': len(devset),
        'n_mel_adapter_params': sum(p.numel() for p in parakeet_asr.mel_adapter.parameters()),
        'n_nemo_adapter_params': sum(p.numel() for p in parakeet_asr.asr.parameters()
                                     if p.requires_grad),
        'mel_adapter_receptive_field': parakeet_asr.mel_adapter.receptive_field,
    })

    warmup_flag = {'active': False}

    def schedule_lr(step):
        """linear warmup -> cosine decay (same intent as NeMo's CosineAnnealing)."""
        if step < FLAGS.parakeet_warmup:
            scale = (step + 1) / FLAGS.parakeet_warmup
        elif FLAGS.parakeet_min_lr_ratio <= 0:
            scale = 1.0
        else:
            prog = (step - FLAGS.parakeet_warmup) / max(1, total_steps - FLAGS.parakeet_warmup)
            prog = min(1.0, max(0.0, prog))
            r = FLAGS.parakeet_min_lr_ratio
            scale = r + (1.0 - r) * 0.5 * (1.0 + math.cos(math.pi * prog))
        for gi, (g, base) in enumerate(zip(optim.param_groups, base_lrs)):
            # gi == 1 is the NeMo adapter group. It is frozen during MelAdapter warmup.
            g['lr'] = 0.0 if (gi == 1 and warmup_flag['active']) else base * scale

    dataloader = torch.utils.data.DataLoader(
        trainset, pin_memory=(device == 'cuda'), collate_fn=trainset.collate_raw,
        num_workers=0, batch_sampler=sampler)

    best_wer = float('inf')
    step = 0

    for epoch_idx in range(FLAGS.parakeet_epochs):
        rnnt_losses, mel_losses = [], []
        optim.zero_grad()

        # --- MelAdapter warmup: set the NeMo adapter lr to 0 for the first N epochs ---
        in_warmup = epoch_idx < FLAGS.mel_warmup_epochs
        warmup_flag['active'] = in_warmup
        if FLAGS.mel_warmup_epochs > 0:
            if epoch_idx == 0:
                logging.info('MelAdapter warmup: NeMo adapter lr=0 for %d epochs',
                             FLAGS.mel_warmup_epochs)
            elif epoch_idx == FLAGS.mel_warmup_epochs:
                logging.info('Warmup finished - starting NeMo adapter training')

        # --- When SpecAugment kicks in ---
        parakeet_asr.spec_augment_enabled = (
            FLAGS.use_spec_augment and epoch_idx >= FLAGS.spec_augment_start_epoch)

        for i, full_batch in enumerate(tqdm.tqdm(dataloader, 'Parakeet train', disable=None)):
            schedule_lr(step)

            # SizeAwareSampler can emit an empty batch (when a single utterance exceeds
            # emg_batch_tokens). Skip it.
            if not full_batch.get('lengths'):
                continue

            for batch in split_batch(full_batch, FLAGS.max_utts_per_batch):
                if FLAGS.joint_finetune:
                    mels, pred, phoneme_pred = predict_mel_batch(transduction_model, batch, device)
                else:
                    with torch.no_grad():
                        mels, pred, phoneme_pred = predict_mel_batch(transduction_model, batch, device)
                    mels = [m.detach() for m in mels]

                texts = batch_texts(batch, trainset)
                if step == 0:
                    logging.info('sample training targets: %s', texts[:3])

                asr_loss = parakeet_asr(mels, texts)
                loss = asr_loss
                rnnt_losses.append(asr_loss.item())

                if FLAGS.joint_finetune and FLAGS.joint_dtw_weight > 0:
                    mel_loss, _ = dtw_loss(pred, phoneme_pred, batch)
                    loss = loss + FLAGS.joint_dtw_weight * mel_loss
                    mel_losses.append(mel_loss.item())

                (loss / FLAGS.grad_accum).backward()
                del loss, asr_loss, mels, pred, phoneme_pred

            if (i + 1) % FLAGS.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(parakeet_asr.trainable_parameters(), 1.0)
                if FLAGS.joint_finetune:
                    torch.nn.utils.clip_grad_norm_(transduction_model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()

            if wandb is not None and step % FLAGS.wandb_log_every == 0 and rnnt_losses:
                log = {
                    'train/rnnt_loss': float(np.mean(rnnt_losses[-FLAGS.wandb_log_every:])),
                    'train/lr_mel_adapter': optim.param_groups[0]['lr'],
                    'train/lr_nemo_adapter': optim.param_groups[1]['lr'],
                    'epoch': epoch_idx + 1,
                }
                if mel_losses:
                    log['train/stage1_loss'] = float(np.mean(mel_losses[-FLAGS.wandb_log_every:]))
                wandb.log(log, step=step)
            step += 1

        epoch_loss = float(np.mean(rnnt_losses))
        msg = f'epoch {epoch_idx+1} - RNNT loss: {epoch_loss:.4f}'
        if mel_losses:
            msg += f' stage-1 ({FLAGS.dtw_loss_type}) loss: {np.mean(mel_losses):.4f}'
        logging.info(msg)

        parakeet_asr.save(os.path.join(FLAGS.output_directory, 'parakeet_asr.pt'))
        if FLAGS.joint_finetune:
            torch.save(transduction_model.state_dict(),
                       os.path.join(FLAGS.output_directory, 'model.pt'))

        epoch_log = {'epoch': epoch_idx + 1, 'train/epoch_rnnt_loss': epoch_loss}

        if FLAGS.eval_every and (epoch_idx + 1) % FLAGS.eval_every == 0:
            wer, cer, targets, predictions = evaluate_dev(
                transduction_model, parakeet_asr, devset, device, FLAGS.eval_max_examples)
            if wer is not None:
                logging.info(f'epoch {epoch_idx+1} - dev WER {wer*100:.3f}% CER {cer*100:.3f}%')
                epoch_log['dev/wer'] = wer * 100
                epoch_log['dev/cer'] = cer * 100

                if wandb is not None:
                    try:
                        table = wandb.Table(columns=['ref', 'hyp'])
                        for t, p in list(zip(targets, predictions))[:20]:
                            table.add_data(t, p)
                        epoch_log['dev/samples'] = table
                    except Exception as e:  # noqa: BLE001
                        logging.debug('Failed to build the wandb table: %s', e)

                if wer < best_wer:
                    best_wer = wer
                    epoch_log['dev/best_wer'] = best_wer * 100
                    parakeet_asr.save(os.path.join(FLAGS.output_directory, 'parakeet_asr_best.pt'))
                    if FLAGS.joint_finetune:
                        torch.save(transduction_model.state_dict(),
                                   os.path.join(FLAGS.output_directory, 'model_best.pt'))

        if wandb is not None:
            wandb.log(epoch_log, step=step)

    logging.info(f'best dev WER: {best_wer*100:.3f}%')
    if wandb is not None:
        wandb.summary['best_dev_wer'] = best_wer * 100
        wandb.finish()


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()