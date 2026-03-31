from torch import nn
from einops.layers.torch import Rearrange
from Models.EEGNet import calculate_outsize
from Models.EEGNet import Conv2dWithConstraint, DenseWithConstraint
import torch
import torch.nn.functional as F
from torchsummary import summary
import torchinfo


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_channels = 62
        self.fs = 128
        self.n_class = 2
        self.dropout_rate = 0.5
        self.t_dropout = 0.5
        self.e_layers = 1
        self.d_model = 16
        self.n_heads = 1
        self.d_ff = 16 * 4
        self.projection_dim = 16  # projection space

        self.CNN_Block = self.cnn_feature_extract_block()
        self.Transformer_Block = self.transformer_feature_extract_block()
        self.BasicBlockOutputSize = calculate_outsize(self.CNN_Block, self.n_channels, self.fs)  # Transformer不改变shape
        self.ClassifierBlock = self.classifier_block(self.projection_dim)
        # position encoding
        self.position = create_1d_absolute_sin_cos_embedding(16, 16)
        self.trick = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Flatten(),
        )
        self.projection_head = nn.Sequential(
            DenseWithConstraint(self.BasicBlockOutputSize, self.projection_dim, max_norm=0.5),
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
            nn.Linear(input_size, self.n_class, bias=True),
            # nn.Softmax(dim=1)
        )
        return Block3

    def forward(self, x):
        x = self.CNN_Block(x)  # (b, 32, 16)
        x = x.squeeze(dim=2)  # (b, seq_len, input_dim)
        x = x.permute(0, 2, 1)
        x = x + self.position.to(x.device)
        x = self.Transformer_Block(x)

        x = self.trick(x)
        x = self.projection_head(x)
        feature = x
        x = self.ClassifierBlock(x)

        return x, F.normalize(feature, p=2, dim=-1)


def create_1d_absolute_sin_cos_embedding(pos_len, dim):
    assert dim % 2 == 0, "wrong dimension!"
    position_emb = torch.zeros(pos_len, dim, dtype=torch.float)
    # i矩阵
    i_matrix = torch.arange(dim//2, dtype=torch.float)
    i_matrix /= dim / 2
    i_matrix = torch.pow(10000, i_matrix)
    i_matrix = 1 / i_matrix
    i_matrix = i_matrix.to(torch.long)
    # pos矩阵
    pos_vec = torch.arange(pos_len).to(torch.long)
    # 矩阵相乘，pos变成列向量，i_matrix变成行向量
    out = pos_vec[:, None] @ i_matrix[None, :]
    # 奇/偶数列
    emb_cos = torch.cos(out)
    emb_sin = torch.sin(out)
    # 赋值
    position_emb[:, 0::2] = emb_sin
    position_emb[:, 1::2] = emb_cos
    return position_emb


if __name__ == '__main__':
    model = Model()
    # x = torch.randn(1, 1, 62, 128)
    # ouput, feature = model(x)
    print(torchinfo.summary(model, (1, 1, 62, 128), device='cpu'))