from torch import nn
from einops.layers.torch import Rearrange
from Models.EEGNet import calculate_outsize
from Models.EEGNet import Conv2dWithConstraint, DenseWithConstraint
import torch.nn.functional as F
from Models.ablation_study.TTMTN import create_1d_absolute_sin_cos_embedding


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_channels = configs.n_channels
        self.fs = configs.fs
        self.n_class = configs.n_class
        self.dropout_rate = configs.dropout_rate
        self.t_dropout = configs.dropout
        self.e_layers = configs.e_layers
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_model * 4

        self.CNN_Block = self.cnn_feature_extract_block()
        self.Transformer_Block = self.transformer_feature_extract_block()
        self.BasicBlockOutputSize = calculate_outsize(self.CNN_Block, self.n_channels, self.fs)  # Transformer不改变shape
        self.ClassifierBlock = self.classifier_block(self.BasicBlockOutputSize)
        # 位置编码
        self.position = create_1d_absolute_sin_cos_embedding(16, 16)
        self.activation = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Flatten(),
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
                max_norm=0.5
            ),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(self.dropout_rate),
        )
        return Block1

    def transformer_feature_extract_block(self):
        Encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.t_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        Block2 = nn.TransformerEncoder(Encoder_layer, num_layers=self.e_layers)
        return Block2

    def classifier_block(self, input_size):
        Block3 = nn.Sequential(
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
        x = x.squeeze(dim=2)  # (b, 16, 16)
        # 绝对位置嵌入
        x = x.permute(0, 2, 1)
        x = x + self.position.cuda()
        x = self.Transformer_Block(x)
        x = self.activation(x)
        feature = x
        x = self.ClassifierBlock(x)

        return x, F.normalize(feature, p=2, dim=-1)
