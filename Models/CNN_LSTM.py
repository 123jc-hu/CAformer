from torch import nn
from einops.layers.torch import Rearrange
from Models.EEGNet import calculate_outsize
import torch
import math


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_channels = configs.n_channels
        self.fs = configs.fs
        self.n_class = configs.n_class
        self.dropout_rate = configs.dropout_rate
        self.e_layers = configs.e_layers
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_model * 4
        self.hidden_dim = 64
        self.num_layers = 1

        self.CNN_Block = self.cnn_feature_extract_block()
        self.LSTM_Block = self.lstm_feature_extract_block()
        self.linear = nn.Linear(self.hidden_dim, self.n_class)
        # self.BasicBlockOutputSize = calculate_outsize(self.CNN_Block, self.n_channels, self.fs)  # Transformer不改变shape
        # self.ClassifierBlock = self.classifier_block(self.BasicBlockOutputSize)

    def cnn_feature_extract_block(self):
        Block1 = nn.Sequential(
            nn.Conv2d(
                1,
                16,
                (1, self.fs // 2),
                stride=(1, 1),
                bias=True,
                padding=(0, self.fs // 4),
            ),
            nn.BatchNorm2d(16),
            nn.Conv2d(
                16,
                32,
                (self.n_channels, 1),
                stride=(1, 1),
                bias=True,
                groups=16
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 8)),
            nn.Dropout(self.dropout_rate),
            Rearrange("b e h w -> b (h w) e")
        )
        return Block1

    def lstm_feature_extract_block(self):
        Block2 = nn.LSTM(32, self.hidden_dim, self.num_layers, batch_first=True)
        return Block2

    # def classifier_block(self, input_size):
    #     Block3 = nn.Sequential(
    #         nn.Flatten(),
    #         nn.Linear(
    #             input_size,
    #             self.n_class,
    #             bias=True
    #         ),
    #         # nn.Softmax(dim=1)
    #     )
    #     return Block3

    def forward(self, x):
        x = self.CNN_Block(x)  # (b, 32, 1, 16)

        x = x.squeeze(dim=2)
        lstm_out, (hidden, cell) = self.LSTM_Block(x)
        last_hidden = hidden[-1]
        logits = self.linear(last_hidden)
        return logits


