import torch.nn.functional as F
import torch.nn as nn
from Models.EEGNet import calculate_outsize


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.n_channels = configs.n_channels
        self.fs = configs.fs
        self.dropout_rate = configs.dropout_rate

        # Block1: 5层dilated conv2d，每层dilated输出维度都是8，kernal是(1,3)，padding=same，dilation分别是(1,2),(1,4),(1,8),(1,16),(1,32)
        self.Block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 3), padding=(0, 2), dilation=(1, 2)),
            nn.Conv2d(8, 8, kernel_size=(1, 3), padding=(0, 4), dilation=(1, 4)),
            nn.Conv2d(8, 8, kernel_size=(1, 3), padding=(0, 8), dilation=(1, 8)),
            nn.Conv2d(8, 8, kernel_size=(1, 3), padding=(0, 16), dilation=(1, 16)),
            nn.Conv2d(8, 8, kernel_size=(1, 3), padding=(0, 32), dilation=(1, 32)),
            nn.BatchNorm2d(8),
            nn.ELU()
        )

        # Block2: 一层conv2d，输出维度是16，kernal是(C, 1)，没有padding，然后接BN层、ELU和dropout
        self.Block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(self.n_channels, 1)),  # input_channels 表示通道数C
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(self.dropout_rate)
        )

        self.BasicBlockOutputSize = calculate_outsize(
            nn.Sequential(self.Block1, self.Block2), self.n_channels, self.fs)

        # 分类层: Flatten+全连接层，输出为2
        self.ClassifierBlock = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.BasicBlockOutputSize, 2)
        )

    def forward(self, x):
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.ClassifierBlock(x)
        return x, None
