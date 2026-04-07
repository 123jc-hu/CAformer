from torch import nn
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class DenseWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


def calculate_outsize(model, channels, samples):
    data = torch.rand(1, 1, channels, samples)
    model.eval()
    out = model(data).shape
    return out.numel()


class Model(nn.Module):
    """CAFormer model used in the paper."""
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

        self.CNN_Block = self.build_case_encoder()
        self.Transformer_Block = self.build_temporal_transformer()
        self.BasicBlockOutputSize = calculate_outsize(self.CNN_Block, self.n_channels, self.fs)  # Transformer不改变shape
        self.ClassifierBlock = self.classifier_block(self.BasicBlockOutputSize)
        # 位置编码
        self.position = create_1d_absolute_sin_cos_embedding(16, 16)
        self.activation = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Flatten(),
        )
        self.projection_head = nn.Sequential(
            DenseWithConstraint(self.BasicBlockOutputSize, self.projection_dim),
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

    def build_case_encoder(self):
        return self.cnn_feature_extract_block()

    @property
    def case_encoder(self):
        return self.CNN_Block

    @property
    def temporal_transformer(self):
        return self.Transformer_Block

    @property
    def classification_head(self):
        return self.ClassifierBlock

    @property
    def temporal_positional_encoding(self):
        return self.position

    @property
    def contrastive_projection_head(self):
        return self.projection_head

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

    def build_temporal_transformer(self):
        return self.transformer_feature_extract_block()

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
        x = x + self.position.to(x.device)
        x = self.Transformer_Block(x)
        x = self.activation(x)
        feature = self.projection_head(x)
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
