from torch import nn
from Models.EEGNet import Conv2dWithConstraint, DenseWithConstraint, calculate_outsize


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_channels = configs.n_channels
        self.fs = configs.fs
        self.n_classes = configs.n_class
        self.dropout_rate = configs.dropout_rate

        self.BasicBlock = self.feature_extract_blocks()
        self.BasicBlockOutputSize = calculate_outsize(self.BasicBlock, self.n_channels, self.fs)
        self.ClassifierBlock = self.classifier_block(self.BasicBlockOutputSize)

    def feature_extract_blocks(self):
        Block1 = nn.Sequential(
            Conv2dWithConstraint(
                1,
                25,
                (1, 5),
                max_norm=2,
                stride=(1, 1),
                bias=False),
            Conv2dWithConstraint(
                25,
                25,
                (self.n_channels, 1),
                max_norm=2,
                stride=(1, 1),
                bias=False),
            nn.BatchNorm2d(25, momentum=0.1, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Dropout2d(p=self.dropout_rate),
        )
        Block2 = nn.Sequential(
            Conv2dWithConstraint(
                25,
                50,
                (1, 5),
                max_norm=2,
                stride=(1, 1),
                bias=False),
            nn.BatchNorm2d(50, momentum=0.1, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Dropout2d(p=self.dropout_rate),
        )
        Block3 = nn.Sequential(
            Conv2dWithConstraint(
                50,
                100,
                (1, 5),
                max_norm=2,
                stride=(1, 1),
                bias=False),
            nn.BatchNorm2d(100, momentum=0.1, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Dropout2d(p=self.dropout_rate),
        )
        Block4 = nn.Sequential(
            Conv2dWithConstraint(
                100,
                200,
                (1, 5),
                max_norm=2,
                stride=(1, 1),
                bias=False),
            nn.BatchNorm2d(200, momentum=0.1, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Dropout2d(p=self.dropout_rate),
        )

        return nn.Sequential(Block1, Block2, Block3, Block4)

    def classifier_block(self, input_size):
        module2 = nn.Sequential(
            nn.Flatten(),
            DenseWithConstraint(
                input_size,
                self.n_classes,
                max_norm=0.5,
                bias=False),
            # nn.Softmax(dim=1)
        )
        return module2

    def forward(self, x):
        x = self.BasicBlock(x)
        x = self.ClassifierBlock(x)
        return x, None
