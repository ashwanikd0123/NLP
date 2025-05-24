import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.tensor, k: torch.tensor, v: torch.tensor, mask: torch.tensor = None):
        scores = torch.bmm(q, k.transpose(1, 2))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        x = F.softmax(scores, dim = 2)
        x = torch.bmm(x, v)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, e: int):
        super().__init__()
        self.q_l = nn.ModuleList([nn.Linear(e, e) for _ in range(h)])
        self.k_l = nn.ModuleList([nn.Linear(e, e) for _ in range(h)])
        self.v_l = nn.ModuleList([nn.Linear(e, e) for _ in range(h)])
        self.attention = Attention()
        self.linear = nn.Linear(e * h, e)

    def forward(self, x: torch.tensor, mask: torch.tensor = None):
        res = None
        for q_lin, k_lin, v_lin in zip(self.q_l, self.k_l, self.v_l):
            cur_q = q_lin(x)
            cur_k = k_lin(x)
            cur_v = v_lin(x)
            cur = self.attention(cur_q, cur_k, cur_v, mask)
            if res is None:
                res = cur
            else:
                res = torch.cat((res, cur), dim = 2)
        res = self.linear(res.contiguous())
        return res

class FeedForward(nn.Module):
    def __init__(self, e: int):
        super().__init__()
        self.l1 = nn.Linear(e, e)
        self.l2 = nn.Linear(e, e)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, e: int, h: int):
        super().__init__()
        self.mha = MultiHeadAttention(h, e)
        self.feed_forward = FeedForward(e)
        self.norm1 = nn.LayerNorm(e)
        self.norm2 = nn.LayerNorm(e)

    def forward(self, x: torch.tensor, mask: torch.tensor):
        old = x.clone()
        x = self.mha(x, mask)
        x = self.norm1(x + old)
        old = x.clone()
        x = self.feed_forward(x)
        x = self.norm2(x + old)
        return x

def pos_encoding(n: int, e: int):
    res = torch.zeros(n, e)
    for i in range(n):
        for j in range(e):
            if j % 2 == 0:
                res[i][j] = math.sin(i / (10000 ** (j / e)))
            else:
                res[i][j] = math.cos(i / (10000 ** ((j - 1) / e)))
    return res

class Bert(nn.Module):
    def __init__(self, embed_size: int, h: int = 6, encoder_count: int = 6):
        super().__init__()
        self.embed_size = embed_size
        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_size, h) for _ in range(encoder_count)])

    def forward(self, x: torch.tensor, mask: torch.tensor = None):
        seq_len = x.size(1)
        pe = pos_encoding(seq_len, self.embed_size).unsqueeze(0).repeat(x.shape[0], 1, 1)
        res = x + pe
        if mask is not None:
            mask = mask.unsqueeze(2).repeat(1, 1, seq_len)
        for encoder in self.encoder_layers:
            res = encoder(res, mask)
        return res