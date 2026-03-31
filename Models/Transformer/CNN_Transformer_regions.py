from torch import nn
from einops.layers.torch import Rearrange
from Models.EEGNet import calculate_outsize, Conv2dWithConstraint
import torch
import ast


def parse_nested_list(nested_list_str):
    """将字符串转换为嵌套列表"""
    return ast.literal_eval(nested_list_str)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_channels_list = parse_nested_list(configs.n_channels_list)
        self.fs = configs.fs
        self.n_class = configs.n_class
        self.dropout_rate = configs.dropout_rate
        self.e_layers = configs.e_layers
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_model * 4

        # Create the CNN and Transformer blocks
        self.blocks = nn.ModuleList(
            [self.multi_cnn_feature_extract_block(len(n_channels)) for n_channels in self.n_channels_list])
        self.Transformer_Block = self.transformer_feature_extract_block()

        # Calculate the output size of the CNN blocks
        self.BasicBlockOutputSize = calculate_outsize(self.blocks[0], len(self.n_channels_list[0]), self.fs)

        # Create the classifier block
        self.ClassifierBlock = self.classifier_block(self.BasicBlockOutputSize*len(self.n_channels_list))

    def multi_cnn_feature_extract_block(self, n_channels):
        return nn.Sequential(
            # Rearrange("b k c t -> b c k t"),
            Conv2dWithConstraint(
                1,
                8,
                (1, self.fs // 2),
                stride=(1, 2),
                bias=True,
                padding=(0, self.fs // 4),
                max_norm=0.5
            ),
            nn.BatchNorm2d(8),
            Conv2dWithConstraint(
                8,
                16,
                (n_channels, 1),
                stride=(1, 1),
                bias=True,
                max_norm=0.5,
            ),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(self.dropout_rate),
        )

    def transformer_feature_extract_block(self):
        Encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        return nn.TransformerEncoder(Encoder_layer, num_layers=self.e_layers)

    def classifier_block(self, input_size):
        return nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Flatten(),
            nn.Linear(input_size, self.n_class, bias=True),
        )

    def forward(self, x):
        x_list = [block(x[:, :, self.n_channels_list[i]]) for i, block in enumerate(self.blocks)]
        x = torch.cat(x_list, dim=1)
        x = x.squeeze(dim=2)  # (b, seq_len, input_dim)
        x = x.permute(1, 0, 2)
        x = self.Transformer_Block(x)
        x = x.permute(1, 0, 2)
        x = self.ClassifierBlock(x)

        return x
