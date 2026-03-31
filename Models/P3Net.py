from torch import nn
from Models.EEGNet import calculate_outsize


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
                32,
                (7, 7),
                stride=(1, 1),
                padding=(3, 3),
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        Block2 = nn.Sequential(
            nn.Conv2d(
                32,
                16,
                (7, 7),
                stride=(1, 1),
                bias=False,
                padding=(3, 3),
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        return nn.Sequential(Block1, Block2)

    def classifier_block(self, input_size):
        Block3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                input_size,
                256,
                bias=False,
            ),
            nn.Dropout(0.1),
            nn.Linear(
                256,
                self.n_classes,
                bias=False,
            ),
            # nn.Softmax(dim=1)
        )
        return Block3

    def forward(self, x):
        x = self.BasicBlock(x)
        x = self.ClassifierBlock(x)
        return x
