from torch import nn
from einops.layers.torch import Rearrange
from Models.EEGNet import calculate_outsize
from Models.EEGNet import Conv2dWithConstraint, DenseWithConstraint
import torch
from torch import Tensor
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import math


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
        self.ClassifierBlock = self.classifier_block(self.projection_dim)

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
        Block2 = TransformerEncoder(self.e_layers, self.d_model)
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
        x = self.projection_head(x)
        feature = x
        x = self.ClassifierBlock(x)

        return x, F.normalize(feature, p=2, dim=-1)


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

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=1,
                 drop_p=0.7,
                 forward_expansion=4,
                 forward_drop_p=0.7):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])