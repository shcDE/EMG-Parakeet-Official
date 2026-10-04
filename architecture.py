import random
import torch
from torch import nn
import torch.nn.functional as F
from absl import flags

from ropetransformer import RoPETransformer 

FLAGS = flags.FLAGS
flags.DEFINE_integer('model_size', 512, 'number of hidden dimensions')
flags.DEFINE_integer('num_layers', 6, 'number of layers')
flags.DEFINE_float('dropout', 0.2, 'dropout')


class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)


class Model(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(8, FLAGS.model_size, stride=2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, stride=2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, stride=2),
        )

        self.w_raw_in = nn.Linear(FLAGS.model_size, FLAGS.model_size)

        self.transformer = RoPETransformer(
            dim=FLAGS.model_size,
            depth=FLAGS.num_layers,
            num_heads=8,
            mlp_ratio=2.0,
            drop_rate=FLAGS.dropout,
            # num_residual_streams=2,
        )

        self.w_out = nn.Linear(FLAGS.model_size, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(FLAGS.model_size, num_aux_outs)

    def forward(self, x_feat, x_raw, session_ids):
        # x_raw shape: (batch, time, electrode) e.g., (32, 1000, 8)

        # Data Augmentation (Random Shift)
        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw = F.pad(x_raw[:, r:, :], (0, 0, 0, r))  # shift left r, zero pad

        x_raw = x_raw.transpose(1, 2)
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1, 2)
        x_raw = self.w_raw_in(x_raw)

        x = x_raw
        x = self.transformer(x)
        out = self.w_out(x)

        if self.has_aux_out:
            return out, self.w_aux(x)
        else:
            return out
