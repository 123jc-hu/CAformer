from torch import nn
from Models.EEGNet import calculate_outsize
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_channels = configs.n_channels
        self.fs = configs.fs
        self.n_classes = configs.n_class

        self.BasicBlock = self.feature_extract_blocks()
        self.BasicBlockOutputSize = calculate_outsize(self.BasicBlock, self.n_channels, self.fs)
        self.ClassifierBlock = self.classifier_block(self.BasicBlockOutputSize)

    def feature_extract_blocks(self):
        Block1 = nn.Sequential(
            nn.Conv2d(
                1,
                96,
                (1, self.n_channels),
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool1d((1, 3), stride=(1, 2)),
            nn.Conv2d(
                96,
                128,
                (1, 6),
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool1d((1, 3), stride=(1, 2)),
            nn.Conv2d(
                128,
                128,
                (1, 6),
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        return Block1

    def classifier_block(self, input_size):
        Block2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                input_size,
                1024,
                bias=False,
            ),
            # nn.Softmax(dim=1)
        )
        return Block2

    def forward(self, x):
        x = self.BasicBlock(x)
        x = self.ClassifierBlock(x)
        return F.normalize(x, p=2, dim=-1), None
