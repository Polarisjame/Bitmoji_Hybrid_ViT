from torch import nn
from torch import Tensor
from einops import rearrange
import torch
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0.1, seq_len: int = 197):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.gate1 = nn.Linear(emb_size//num_heads, emb_size//num_heads)
        self.gate2 = nn.Linear(emb_size//num_heads, emb_size//num_heads)
        self.gate3 = nn.Linear(emb_size//num_heads, 1)
        self.scaling = (self.emb_size // num_heads) ** -0.5  # emb_size = embeddingsize * num_heads
        self.W = nn.Parameter(torch.randn(seq_len,seq_len))
        self.Norm = nn.LayerNorm([seq_len,seq_len])
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        x0 = rearrange(x, "b n (h d) -> b h n d", h=self.num_heads)
        # w = self.gate3(torch.mul(torch.mul(torch.max(x0, axis=-2)[0], torch.max(self.gate2(x0), axis=-2)[0]),
        #                          torch.mean(x0, axis=-2)))
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        att = F.softmax(energy * self.scaling, dim=-1)
        att = self.Norm(self.W * att)
        # att = torch.einsum('bhqk,bhv -> bhqk', att, w)
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
