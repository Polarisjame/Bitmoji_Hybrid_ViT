import torch
from torch import nn
from torch import Tensor
import copy

from utils.PE import PatchEmbedding
from utils.Attention import MultiHeadAttention
from utils.ResidualNet import Res34


# 克隆N层网络
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 归一化层
class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 残差链接层
class ResidualAdd(nn.Module):

    def __init__(self, size, dropout, rezero):
        super(ResidualAdd, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# 前馈全连接层
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model, expansion: int = 4, dropout: float = 0.1):
        super(FeedForwardBlock, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model * expansion)
        self.w_2 = nn.Linear(d_model * expansion, d_model)
        self.dropout = nn.Dropout(dropout)
        self.GELU = nn.GELU()

    def forward(self, x):
        return self.dropout(self.w_2(self.dropout(self.GELU(self.w_1(x)))))


# 单个Encoder层
class EncoderLayer(nn.Module):

    def __init__(self, size: int, self_attn, feed_forward, dropout: float = 0.1, rezero=True):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualAdd(size, dropout, rezero), 2)
        self.size = size

    def forward(self, x, mask: Tensor = None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, mask))
        return self.sublayer[1](x, self.feed_forward)


# 分类器
class Classifaction(nn.Module):
    # b*n*e -> b*n_classes
    def __init__(self, emb_size: int = 768, n_classes: int = 2):
        super(Classifaction, self).__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        # x = Reduce('b n e -> b e', 'mean')(x)
        x = x[:, 0]  # get CLS token
        x = self.norm(x)
        return self.linear(x)


class VIT(nn.Module):
    def __init__(self, args, in_channels: int = 3, patches: int = 16, emb_size: int = 768, num_head: int = 8,
                 img_size: int = 224, depth: int = 12, n_classes: int = 2):
        super(VIT, self).__init__()
        if args.in_channels:
            in_channels = args.in_channels
        seq_len = img_size//patches ** 2 + 1
        self.depth = depth
        self.embed = PatchEmbedding(in_channels, patches, emb_size, seq_len)
        self.dropout = args.dropout
        seq_len = (img_size // patches) ** 2 + 1
        self.encodelayer = EncoderLayer(emb_size, MultiHeadAttention(emb_size, num_head, self.dropout, seq_len),
                                        FeedForwardBlock(emb_size, 4, dropout=self.dropout), dropout=self.dropout,
                                        rezero=args.re_zero)
        self.encodes = clones(self.encodelayer, depth)
        self.cretify = Classifaction(emb_size, n_classes)
        self.hybrid = args.use_hybrid
        if args.use_hybrid:
            self.embed = PatchEmbedding(args.resnet_out_channels, 1, emb_size, seq_len)
            self.resnet = Res34(args, in_channels, args.resnet_out_channels)

    def forward(self, x, mask: Tensor = None):
        if self.hybrid:
            x = self.resnet(x)
        x = self.embed(x)
        for encode in self.encodes:
            x = encode(x, mask)
        return self.cretify(x)
