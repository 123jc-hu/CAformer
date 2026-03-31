from torch import nn
from einops.layers.torch import Rearrange
from Models.EEGNet import calculate_outsize
from Models.EEGNet import Conv2dWithConstraint, DenseWithConstraint
import torch


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
        self.projection_dim = configs.projection_dim

        # self.layernorm = nn.LayerNorm(self.d_model)

        self.CNN_Block = self.cnn_feature_extract_block()
        self.Transformer_Block = self.transformer_feature_extract_block()
        self.BasicBlockOutputSize = calculate_outsize(self.CNN_Block, self.n_channels, self.fs)  # Transformer不改变shape
        self.ClassifierBlock = self.classifier_block(self.BasicBlockOutputSize)
        # 可训练的位置编码
        # self.positional_encoding = PositionalEncoding(configs.d_model)
        # self.pos_encoder = PositionalEncoding(16, 16)
        self.position = create_1d_absolute_sin_cos_embedding(16, 16)
        self.trick = nn.Sequential(
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

            DenseWithConstraint(
                input_size,
                self.n_class,
                bias=True,
                # max_norm=0.5
            ),
            # nn.Softmax(dim=1)
        )
        return Block3

    def forward(self, x):
        x = self.CNN_Block(x)  # (b, 32, 16)
        x = x.squeeze(dim=2)  # (b, seq_len, input_dim)
        # 绝对位置嵌入
        # x = self.pos_encoder(x)
        # x = self.positional_encoding(x)
        x = x.permute(0, 2, 1)
        x = x + self.position.cuda()
        x = self.Transformer_Block(x)
        # x = x.permute(1, 0, 2)

        # x = x.unsqueeze(dim=2)
        # x = self.layernorm(x)

        x = self.trick(x)
        x = self.ClassifierBlock(x)

        return x, None


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_position=10000):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = input_dim
        self.positional_embeddings = nn.Parameter(torch.randn(max_position, input_dim))

    def forward(self, x):
        seq_len = x.size(0)  # 注意这里使用size(0)获取序列长度
        positional_encoding = self.positional_embeddings[:seq_len, :]
        output = x + positional_encoding.unsqueeze(1)  # 对应位置添加位置编码
        return output



# 1d绝对sin_cos编码
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

