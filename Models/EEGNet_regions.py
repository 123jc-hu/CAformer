import torch
from torch import nn
from Models.EEGNet import Conv2dWithConstraint, DenseWithConstraint, calculate_outsize
from Models.Transformer.CNN_Transformer_regions import parse_nested_list


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.n_channels_list = parse_nested_list(configs.n_channels_list)
        self.fs = configs.fs
        self.n_classes = configs.n_class
        self.kernel_length = self.fs // 2
        self.kernel_length2 = self.fs // 8
        self.dropout_rate = configs.dropout_rate

        self.BasicBlock = self.Block1()
        self.blocks = nn.ModuleList(
            [self.Block2(len(n_channels)) for n_channels in self.n_channels_list])
        self.BasicBlock3 = self.Block3()
        # self.BasicBlockOutputSize = calculate_outsize(self.BasicBlock, self.n_channels, self.fs)
        self.ClassifierBlock = self.classifier_block(64*8)

    def Block1(self):
        return nn.Sequential(
            nn.Conv2d(
                1,
                self.F1,
                (1, self.kernel_length),
                stride=1,
                bias=False,
                padding=(0, self.kernel_length // 2),
            ),
            nn.BatchNorm2d(self.F1),
        )

    def Block2(self, n_channels):
        return nn.Sequential(
            # DepthwiseConv2D =======================
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (n_channels, 1),
                max_norm=1,
                stride=1,
                bias=False,
                # groups=self.F1,
            ),
            # ========================================
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(p=self.dropout_rate)
        )

    def Block3(self):
        return nn.Sequential(
            # SeparableConv2D =======================
            nn.Conv2d(
                self.F1 * self.D * 8,
                self.F1 * self.D * 8,
                (1, self.kernel_length2),
                stride=1,
                bias=False,
                padding=(0, self.kernel_length2 // 2),
                groups=self.F1 * self.D * 8
            ),
            nn.Conv2d(
                self.F1 * self.D * 8,
                self.F2 * 8,
                (1, 1),
                stride=1,
                bias=False,
            ),
            # ========================================
            nn.BatchNorm2d(self.F2 * 8),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=self.dropout_rate)
        )

    def classifier_block(self, input_size):
        return nn.Sequential(
            nn.Flatten(),
            DenseWithConstraint(
                input_size,
                self.n_classes,
                bias=False,
                max_norm=0.25),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.BasicBlock(x)
        x_list = [block(x[:, :, self.n_channels_list[i]]) for i, block in enumerate(self.blocks)]
        x = torch.cat(x_list, dim=1)
        x = self.BasicBlock3(x)
        x = self.ClassifierBlock(x)
        return x
