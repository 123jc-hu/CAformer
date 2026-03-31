from torch import nn
from einops.layers.torch import Rearrange
from Models.EEGNet import calculate_outsize
from Models.EEGNet import Conv2dWithConstraint, DenseWithConstraint
import torch
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_channels = configs.n_channels
        self.fs = configs.fs
        self.n_class = configs.n_class
        self.dropout_rate = configs.dropout_rate
        self.projection_dim = configs.projection_dim

        self.CNN_Block = self.cnn_feature_extract_block()
        self.BasicBlockOutputSize = calculate_outsize(self.CNN_Block, self.n_channels, self.fs)  # Transformer不改变shape
        self.ClassifierBlock = self.classifier_block(self.BasicBlockOutputSize)
        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.BasicBlockOutputSize, self.projection_dim),
            nn.GELU(),
            # nn.Flatten()
        )

    def cnn_feature_extract_block(self):
        Block1 = nn.Sequential(
            Rearrange("b k c t -> b c k t"),
            Conv2dWithConstraint(
                self.n_channels,
                16,
                (1, self.fs // 2),
                stride=(1, 2),
                bias=True,
                padding=(0, self.fs // 4),
                # max_norm=0.5
            ),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(self.dropout_rate),
        )
        return Block1

    def classifier_block(self, input_size):
        Block3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                input_size,
                self.n_class,
                bias=True,
            ),
            # nn.Softmax(dim=1)
        )
        return Block3

    def forward(self, x):
        x = self.CNN_Block(x)  # (b, 16, 1, 16)
        feature = self.projection_head(x)
        x = self.ClassifierBlock(x)

        return x, F.normalize(feature, p=2, dim=-1)
