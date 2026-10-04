import logging
import random

import torch
from torch import nn
import torch.nn.functional as F

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('parakeet_model_name', 'nvidia/parakeet-tdt-1.1b',
                    'NeMo pretrained model name (or path to a local .nemo file)')
flags.DEFINE_string('parakeet_checkpoint', None,
                    'Path to a .pt file holding trained mel adapter + NeMo adapter weights')

# ---- MelAdapter (mel -> parakeet feature space) ----
flags.DEFINE_integer('adapter_hidden', 256, 'MelAdapter hidden dim')
flags.DEFINE_integer('adapter_layers', 4,
                     'Total number of MelAdapter conv layers (1 dense stem + N-1 body blocks)')
flags.DEFINE_integer('adapter_kernel', 9,
                     'MelAdapter conv kernel size (must be odd). Separable convolutions make '
                     'large kernels nearly free, so this can be widened to grow the temporal '
                     'receptive field.')
flags.DEFINE_boolean('adapter_use_tcs', True,
                     'Use 1D time-channel separable convolutions in the MelAdapter body. '
                     'Set to False for the dense-convolution ablation baseline.')
flags.DEFINE_float('adapter_dropout', 0.1, 'MelAdapter dropout')
flags.DEFINE_string('adapter_out_norm', 'per_feature',
                    "MelAdapter output normalization: 'per_feature' (matches the NeMo "
                    "preprocessor, recommended) | 'layer' | 'none'")
flags.DEFINE_float('emg_mel_frame_rate', 22050.0 / 256.0,
                   'Frame rate (Hz) of the mel produced by the transduction model')

# ---- NeMo encoder adapter (asymmetric) ----
flags.DEFINE_string('nemo_adapter_name', 'emg-finetune', 'Adapter name')
flags.DEFINE_integer('adapter_lower_dim', 8, 'Adapter bottleneck dim for the lower half of the layers')
flags.DEFINE_integer('adapter_upper_dim', 64, 'Adapter bottleneck dim for the upper half of the layers')
flags.DEFINE_float('nemo_adapter_dropout', 0.1, 'NeMo LinearAdapter dropout')
flags.DEFINE_boolean('freeze_bn_stats', True,
                     'Freeze the encoder BatchNorm running stats (prevents contamination '
                     'by EMG-derived mel statistics)')
flags.DEFINE_boolean('use_spec_augment', False, 'Apply SpecAugment during training')
flags.DEFINE_integer('spec_freq_masks', 2, 'SpecAugment freq masks')
flags.DEFINE_integer('spec_time_masks', 10, 'SpecAugment time masks')


def differentiable_spec_augment(x, lengths, freq_masks, time_masks,
                                freq_width=27, time_width=0.05):
    """Differentiable SpecAugment.

    NeMo's SpectrogramAugmentation.forward is wrapped in torch.no_grad() (it assumes
    it operates on preprocessor output), so no gradient reaches the upstream
    MelAdapter. Here the augmentation is implemented as a multiplication by a 0/1
    mask, which preserves the gradient path.

    x: (B, D, T)
    """
    if freq_masks <= 0 and time_masks <= 0:
        return x
    B, D, T = x.shape
    mask = torch.ones_like(x, requires_grad=False)
    for i in range(B):
        L = max(1, int(lengths[i]))
        for _ in range(freq_masks):
            w = random.randint(0, freq_width)
            if w > 0 and D > w:
                f0 = random.randint(0, D - w)
                mask[i, f0:f0 + w, :] = 0.0
        tw = int(time_width * L) if isinstance(time_width, float) else int(time_width)
        for _ in range(time_masks):
            w = random.randint(0, max(1, tw))
            if w > 0 and L > w:
                t0 = random.randint(0, L - w)
                mask[i, :, t0:t0 + w] = 0.0
    return x * mask


# =========================================================================
# Convolution blocks
# =========================================================================

class ConvBlock(nn.Module):
    """Dense 1D convolution block: conv -> GroupNorm -> GELU -> dropout.

    Used for the channel-expansion stem (where separable factorization saves
    almost nothing because the input channel count is small), and as the
    dense-convolution ablation baseline for the adapter body.

    GroupNorm(1, C) is used instead of BatchNorm because the input is variable
    length and zero padded; batch statistics would be contaminated by padding,
    and BatchNorm would also conflict with the freeze_bn_stats logic applied to
    the frozen encoder.

    Parameters: C_in * C_out * K + C_out (conv) + 2 * C_out (norm)
    """

    def __init__(self, c_in, c_out, kernel_size, dropout, residual=False):
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel_size must be odd for symmetric padding'
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(1, c_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = residual and (c_in == c_out)

    def forward(self, x):
        y = self.drop(self.act(self.norm(self.conv(x))))
        return x + y if self.residual else y


class TCSConv(nn.Module):
    """1D time-channel separable convolution block (QuartzNet / Citrinet style).

    A standard 1D convolution is factorized into:
      - depthwise: kernel of length K applied to each channel independently,
                   across K time frames                      -> C_in * K params
      - pointwise: 1x1 convolution mixing all channels at
                   each time frame independently             -> C_in * C_out params

    Cost drops from C_in * C_out * K to C_in * K + C_in * C_out. At C=256 and K=9
    that is 590k -> 68k parameters per layer (~8.6x), and the depthwise term scales
    linearly in K rather than multiplicatively, so widening the kernel to grow the
    receptive field costs almost nothing.

    A residual connection is used whenever the block preserves the channel count,
    matching the residual structure of QuartzNet blocks.
    """

    def __init__(self, c_in, c_out, kernel_size, dropout):
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel_size must be odd for symmetric padding'
        self.depthwise = nn.Conv1d(c_in, c_in, kernel_size,
                                   padding=kernel_size // 2, groups=c_in, bias=False)
        self.pointwise = nn.Conv1d(c_in, c_out, kernel_size=1)
        self.norm = nn.GroupNorm(1, c_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = (c_in == c_out)

    def forward(self, x):
        y = self.drop(self.act(self.norm(self.pointwise(self.depthwise(x)))))
        return x + y if self.residual else y


# =========================================================================
# MelAdapter
# =========================================================================

class MelAdapter(nn.Module):
    """(T_mel, num_mel) -> (T_feat, parakeet_feat_dim)

    The time axis is interpolated per sample (so that padding never leaks into a
    neighbouring utterance), while the channel axis is handled by convolutions
    plus a final linear projection.

    Output normalization mimics the NeMo preprocessor's normalize='per_feature'
    (each mel bin is normalized by its own mean/std over the time axis, per
    utterance).
    """

    def __init__(self, num_mel, out_dim, hidden=256, num_layers=4, kernel_size=9,
                 dropout=0.1, out_norm='per_feature', use_tcs=True):
        super().__init__()
        assert num_layers >= 1, 'num_layers must be at least 1'
        self.num_mel = num_mel
        self.out_dim = out_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.use_tcs = use_tcs
        self.out_norm_type = out_norm

        self.in_norm = nn.LayerNorm(num_mel)

        # Stem: dense convolution for the num_mel -> hidden channel expansion.
        # Separable factorization is not worth it here: with a small c_in the
        # depthwise term dominates and the savings are negligible.
        self.stem = ConvBlock(num_mel, hidden, kernel_size, dropout, residual=False)

        # Body: separable (or dense, for the ablation) residual blocks at constant width.
        body = []
        for _ in range(num_layers - 1):
            if use_tcs:
                body.append(TCSConv(hidden, hidden, kernel_size, dropout))
            else:
                body.append(ConvBlock(hidden, hidden, kernel_size, dropout, residual=True))
        self.convs = nn.Sequential(*body)

        self.proj = nn.Linear(hidden, out_dim)
        if out_norm == 'layer':
            self.out_ln = nn.LayerNorm(out_dim)
        else:
            self.out_ln = None
        self.out_gain = nn.Parameter(torch.ones(()))
        self.out_bias = nn.Parameter(torch.zeros(()))

    # ---------------------------------------------------------------- info
    @property
    def receptive_field(self):
        """Receptive field in mel frames (all layers use stride 1, dilation 1)."""
        return 1 + (self.kernel_size - 1) * self.num_layers

    def receptive_field_ms(self, frame_rate):
        return 1000.0 * self.receptive_field / frame_rate

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------- forward
    @staticmethod
    def _per_feature_norm(x, lengths, eps=1e-5):
        """x: (B, D, T). Normalize each sample/channel by the mean/std of its valid span."""
        out = torch.zeros_like(x)
        for i, L in enumerate(lengths):
            xi = x[i, :, :L]
            m = xi.mean(dim=1, keepdim=True)
            s = xi.std(dim=1, keepdim=True)
            out[i, :, :L] = (xi - m) / (s + eps)
        return out

    def forward(self, mels, feat_lengths):
        """
        mels: list of (T_i, num_mel) tensors (all on the same device)
        feat_lengths: list of int, the number of parakeet feature frames each
                      sample should have
        return: (B, out_dim, T_feat_max)  <- the NeMo encoder expects (B, D, T)
        """
        device = mels[0].device
        B = len(mels)
        mel_lengths = [m.shape[0] for m in mels]
        T_mel = max(mel_lengths)

        padded = torch.zeros(B, T_mel, self.num_mel, device=device, dtype=mels[0].dtype)
        for i, m in enumerate(mels):
            padded[i, :m.shape[0]] = m

        x = self.in_norm(padded).transpose(1, 2)            # (B, num_mel, T_mel)
        x = self.stem(x)                                     # (B, hidden, T_mel)
        x = self.convs(x)                                    # (B, hidden, T_mel)

        T_feat = max(feat_lengths)
        out = x.new_zeros(B, x.shape[1], T_feat)
        for i, (L, E) in enumerate(zip(mel_lengths, feat_lengths)):
            xi = x[i:i + 1, :, :L]
            xi = F.interpolate(xi, size=E, mode='linear', align_corners=False)
            out[i:i + 1, :, :E] = xi

        out = out.transpose(1, 2)                            # (B, T_feat, hidden)
        out = self.proj(out)
        if self.out_ln is not None:
            out = self.out_ln(out)
        out = out.transpose(1, 2)                            # (B, out_dim, T_feat)

        if self.out_norm_type == 'per_feature':
            out = self._per_feature_norm(out, feat_lengths)

        return out * self.out_gain + self.out_bias


# =========================================================================
# ParakeetMelASR
# =========================================================================

class ParakeetMelASR(nn.Module):
    """Parakeet-TDT ASR that consumes mel spectrograms directly."""

    def __init__(self, num_mel_features, model_name=None, device='cuda'):
        super().__init__()
        model_name = model_name or FLAGS.parakeet_model_name
        self.device_ = device

        import nemo.collections.asr as nemo_asr  # noqa: F401  (import registers NeMo classes)
        from nemo.collections.asr.models import ASRModel
        from nemo.core import adapter_mixins
        from omegaconf import open_dict

        # ---- swap the encoder target for its adapter-capable class, then load ----
        is_local = model_name.endswith('.nemo')

        def _patch_cfg(cfg):
            with open_dict(cfg):
                meta = adapter_mixins.get_registered_adapter(cfg.encoder._target_)
                if meta is not None:
                    cfg.encoder._target_ = meta.adapter_class_path
                    logging.info('encoder target -> %s', meta.adapter_class_path)
                else:
                    logging.warning('No adapter class found for encoder (%s).',
                                    cfg.encoder._target_)
            return cfg

        if is_local:
            cfg = ASRModel.restore_from(model_name, return_config=True)
            cfg = _patch_cfg(cfg)
            self.asr = ASRModel.restore_from(model_name, override_config_path=cfg,
                                             map_location=device)
        else:
            cfg = ASRModel.from_pretrained(model_name, return_config=True)
            cfg = _patch_cfg(cfg)
            self.asr = ASRModel.from_pretrained(model_name, override_config_path=cfg,
                                                map_location=device)
        self.asr = self.asr.to(device)

        # ---- read the feature specification ----
        pre = self.asr.cfg.preprocessor
        self.feat_dim = int(pre.get('features', 128))
        sr = float(pre.get('sample_rate', 16000))
        stride = float(pre.get('window_stride', 0.01))
        self.parakeet_frame_rate = 1.0 / stride          # typically 100.0 Hz
        self.encoder_dim = int(self.asr.cfg.encoder.get('d_model', 1024))

        logging.info('Parakeet feature spec: feat_dim=%d, frame_rate=%.2fHz, '
                     'encoder d_model=%d, sample_rate=%.0f',
                     self.feat_dim, self.parakeet_frame_rate, self.encoder_dim, sr)

        # ---- SpecAugment ----
        # NeMo's SpectrogramAugmentation runs under no_grad, so no gradient would
        # reach the MelAdapter. differentiable_spec_augment() is used instead,
        # applied inside encode_mels().
        self.asr.spec_augmentation = None

        # ---- disable the joint's fused loss/wer (loss is computed explicitly) ----
        if hasattr(self.asr, 'joint') and hasattr(self.asr.joint, 'set_fuse_loss_wer'):
            self.asr.joint.set_fuse_loss_wer(False)

        # ---- greedy decoding (CUDA graph disabled) ----
        self._set_greedy_decoding()

        # ---- inject the NeMo encoder adapters ----
        self._add_encoder_adapters()
        # Modules created by add_adapter() are allocated on CPU, so move them back.
        self.asr = self.asr.to(device)

        # ---- configure what is trainable ----
        self.asr.freeze()
        self.asr.train()
        self.asr.unfreeze_enabled_adapters()
        if FLAGS.freeze_bn_stats:
            self._freeze_bn_stats()

        # ---- MelAdapter ----
        self.mel_adapter = MelAdapter(
            num_mel_features, self.feat_dim,
            hidden=FLAGS.adapter_hidden,
            num_layers=FLAGS.adapter_layers,
            kernel_size=FLAGS.adapter_kernel,
            dropout=FLAGS.adapter_dropout,
            out_norm=FLAGS.adapter_out_norm,
            use_tcs=FLAGS.adapter_use_tcs).to(device).float()

        logging.info('MelAdapter: %s conv, hidden=%d, layers=%d, kernel=%d, '
                     'receptive field=%d frames (%.0f ms @ %.2f Hz), params=%.3fM',
                     'separable (TCS)' if FLAGS.adapter_use_tcs else 'dense',
                     FLAGS.adapter_hidden, FLAGS.adapter_layers, FLAGS.adapter_kernel,
                     self.mel_adapter.receptive_field,
                     self.mel_adapter.receptive_field_ms(FLAGS.emg_mel_frame_rate),
                     FLAGS.emg_mel_frame_rate,
                     self.mel_adapter.num_parameters() / 1e6)

        self._log_param_counts()

    # ------------------------------------------------------------- setup
    def _set_greedy_decoding(self):
        from omegaconf import OmegaConf
        try:
            cfg = OmegaConf.create({
                'strategy': 'greedy_batch',
                'greedy': {'use_cuda_graph_decoder': False},
            })
            self.asr.change_decoding_strategy(cfg)
        except Exception as e:  # noqa: BLE001
            logging.warning('Failed to set the decoding strategy (%s). Using the default.', e)

    def _add_encoder_adapters(self):
        """Apply the asymmetric strategy from nemo_config.yaml verbatim."""
        from omegaconf import OmegaConf

        lower, upper = FLAGS.adapter_lower_dim, FLAGS.adapter_upper_dim
        name = FLAGS.nemo_adapter_name

        def _cfg(dim):
            return OmegaConf.create({
                '_target_': 'nemo.collections.common.parts.adapter_modules.LinearAdapter',
                'in_features': self.encoder_dim,
                'dim': dim,
                'activation': 'swish',
                'norm_position': 'pre',
                'dropout': FLAGS.nemo_adapter_dropout,
                'adapter_strategy': {
                    '_target_': 'nemo.core.classes.mixins.adapter_mixin_strategies.'
                                'ResidualAddAdapterStrategy',
                    'stochastic_depth': 0.0,
                    'l2_lambda': 0.0,
                },
            })

        layers = self._encoder_layers()
        n = len(layers)

        if lower == upper:
            uni = f'{name}_uniform'
            self.asr.add_adapter(name=f'encoder:{uni}', cfg=_cfg(lower))
            assert self.asr.is_adapter_available()
            self.asr.set_enabled_adapters(enabled=False)
            for layer in layers:
                if hasattr(layer, 'set_enabled_adapters'):
                    layer.set_enabled_adapters(name=uni, enabled=True)
            self.adapter_names = [uni]
            logging.info('Uniform adapter: dim=%d, %d layers', lower, n)
        else:
            lo, up = f'{name}_lower', f'{name}_upper'
            self.asr.add_adapter(name=f'encoder:{lo}', cfg=_cfg(lower))
            self.asr.add_adapter(name=f'encoder:{up}', cfg=_cfg(upper))
            assert self.asr.is_adapter_available()
            self.asr.set_enabled_adapters(enabled=False)
            split = n // 2
            for i, layer in enumerate(layers):
                if hasattr(layer, 'set_enabled_adapters'):
                    layer.set_enabled_adapters(name=lo if i < split else up, enabled=True)
            self.adapter_names = [lo, up]
            logging.info('Asymmetric adapter: layers 0-%d dim=%d / %d-%d dim=%d',
                         split - 1, lower, split, n - 1, upper)

    def _encoder_layers(self):
        enc = self.asr.encoder
        if hasattr(enc, 'layers'):
            return enc.layers
        if hasattr(enc, 'encoder') and hasattr(enc.encoder, 'layers'):
            return enc.encoder.layers
        raise AttributeError('Could not locate the encoder layers.')

    def _freeze_bn_stats(self):
        n = 0
        for m in self.asr.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()
                m.track_running_stats = False
                n += 1
        if n:
            logging.info('Froze BatchNorm running stats: %d modules', n)

    def _log_param_counts(self):
        n_ad = sum(p.numel() for p in self.asr.parameters() if p.requires_grad)
        n_mel = sum(p.numel() for p in self.mel_adapter.parameters())
        n_tot = sum(p.numel() for p in self.asr.parameters())
        logging.info('Parakeet trainable: NeMo adapter %.2fM + mel adapter %.2fM '
                     '(base %.1fM, %.2f%% of the model is trained)',
                     n_ad / 1e6, n_mel / 1e6, n_tot / 1e6,
                     100.0 * (n_ad + n_mel) / n_tot)

    def trainable_parameters(self):
        return list(self.mel_adapter.parameters()) + \
               [p for p in self.asr.parameters() if p.requires_grad]

    def train(self, mode=True):
        super().train(mode)
        self.asr.train(mode)
        self.mel_adapter.train(mode)
        if mode and FLAGS.freeze_bn_stats:
            self._freeze_bn_stats()
        return self

    # ------------------------------------------------------------- core
    def num_feature_frames(self, mel_len):
        r = self.parakeet_frame_rate / FLAGS.emg_mel_frame_rate
        return max(8, int(round(mel_len * r)))

    def encode_mels(self, mels):
        """mels: list of (T_i, num_mel) -> (encoded, encoded_len)

        encoded: (B, D_enc, T_enc)  /  encoded_len: (B,)
        """
        mels = [m.to(self.device_).float() for m in mels]
        feat_lengths = [self.num_feature_frames(m.shape[0]) for m in mels]
        feats = self.mel_adapter(mels, feat_lengths)         # (B, feat_dim, T)
        lens = torch.tensor(feat_lengths, device=self.device_, dtype=torch.long)

        if FLAGS.use_spec_augment and self.training:
            feats = differentiable_spec_augment(
                feats, feat_lengths,
                freq_masks=FLAGS.spec_freq_masks,
                time_masks=FLAGS.spec_time_masks,
                freq_width=27, time_width=0.05)

        encoded, encoded_len = self.asr.encoder(audio_signal=feats, length=lens)
        return encoded, encoded_len

    def _tokenize(self, texts):
        ids = [self.asr.tokenizer.text_to_ids(t) for t in texts]
        lens = torch.tensor([len(x) for x in ids], device=self.device_, dtype=torch.long)
        maxlen = max(1, int(lens.max().item()))
        out = torch.zeros(len(ids), maxlen, device=self.device_, dtype=torch.long)
        for i, x in enumerate(ids):
            if x:
                out[i, :len(x)] = torch.tensor(x, device=self.device_, dtype=torch.long)
        return out, lens

    def forward(self, mels, texts):
        """TDT/RNNT loss. mels: list of (T, num_mel), texts: list of str"""
        encoded, encoded_len = self.encode_mels(mels)
        targets, target_len = self._tokenize(texts)

        decoder_out, _, _ = self.asr.decoder(targets=targets, target_length=target_len)
        joint_out = self.asr.joint(encoder_outputs=encoded, decoder_outputs=decoder_out)

        loss = self.asr.loss(log_probs=joint_out,
                             targets=targets,
                             input_lengths=encoded_len,
                             target_lengths=target_len)
        return loss

    @torch.no_grad()
    def transcribe(self, mels, **kwargs):
        """mels: list of (T, num_mel) -> list of str"""
        was_training = self.training
        self.eval()

        encoded, encoded_len = self.encode_mels(mels)
        out = self.asr.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False)

        # Depending on the NeMo version this is list[str], list[Hypothesis],
        # or a (best, all) tuple.
        if isinstance(out, tuple):
            out = out[0]
        texts = []
        for h in out:
            if isinstance(h, str):
                texts.append(h)
            else:
                texts.append(getattr(h, 'text', str(h)))

        if was_training:
            self.train()
        return [t.strip() for t in texts]

    # -------------------------------------------------------------- i/o
    def save(self, path):
        torch.save({
            'mel_adapter': self.mel_adapter.state_dict(),
            'nemo_adapter': {n: p.detach().cpu()
                             for n, p in self.asr.named_parameters() if p.requires_grad},
            'meta': {
                'model_name': FLAGS.parakeet_model_name,
                'adapter_hidden': FLAGS.adapter_hidden,
                'adapter_layers': FLAGS.adapter_layers,
                'adapter_kernel': FLAGS.adapter_kernel,
                'adapter_use_tcs': FLAGS.adapter_use_tcs,
                'adapter_lower_dim': FLAGS.adapter_lower_dim,
                'adapter_upper_dim': FLAGS.adapter_upper_dim,
                'adapter_out_norm': FLAGS.adapter_out_norm,
                'emg_mel_frame_rate': FLAGS.emg_mel_frame_rate,
                'feat_dim': self.feat_dim,
                'num_mel_features': self.mel_adapter.num_mel,
            },
        }, path)

    def load(self, path, strict=True):
        state = torch.load(path, map_location='cpu')

        # Warn early if the checkpoint was trained with a different MelAdapter shape:
        # a mismatch here produces confusing load errors further down.
        meta = state.get('meta', {})
        for key, current in (('adapter_hidden', FLAGS.adapter_hidden),
                             ('adapter_layers', FLAGS.adapter_layers),
                             ('adapter_kernel', FLAGS.adapter_kernel),
                             ('adapter_use_tcs', FLAGS.adapter_use_tcs)):
            saved = meta.get(key)
            if saved is not None and saved != current:
                logging.warning('Checkpoint %s=%s but the current flag is %s.',
                                key, saved, current)

        self.mel_adapter.load_state_dict(state['mel_adapter'], strict=strict)
        res = self.asr.load_state_dict(state['nemo_adapter'], strict=False)
        n_unexpected = len(getattr(res, 'unexpected_keys', []))
        logging.info('Loaded parakeet checkpoint from %s (unexpected keys: %d)',
                     path, n_unexpected)
        if n_unexpected:
            logging.warning('unexpected keys is non-zero. Check that the adapter dim flags '
                            'match the ones used at training time '
                            '(--adapter_lower_dim / --adapter_upper_dim).')
        self.mel_adapter.to(self.device_)
        self.asr = self.asr.to(self.device_)
        return self


def build_parakeet_asr(num_mel_features, device='cuda', checkpoint=None):
    asr = ParakeetMelASR(num_mel_features, device=device)
    ckpt = checkpoint if checkpoint is not None else FLAGS.parakeet_checkpoint
    if ckpt is not None:
        asr.load(ckpt)
    return asr