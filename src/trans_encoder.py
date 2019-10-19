#!/usr/bin/env python
import torch.nn as nn

from multi_head_attn import MultiHeadAttn
from layer_norm import LayerNorm
from feed_forward import FeedForward

class TransEncoder(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super(TransEncoder, self).__init__()
        self.multi_head_attn = MultiHeadAttn(d_model, num_heads=8)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask):
        attn_x = self.multi_head_attn(query=x, key=x, value=x, mask=mask)
        x = self.norm1(x + self.dropout1(attn_x))
        return self.norm2(x + self.dropout2(self.feed_forward(x)))