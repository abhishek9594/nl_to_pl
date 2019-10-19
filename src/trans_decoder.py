#!/usr/bin/env python
import torch.nn as nn

from multi_head_attn import MultiHeadAttn
from layer_norm import LayerNorm
from feed_forward import FeedForward

class TransDecoder(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super(TransDecoder, self).__init__()
        self.multi_head_masked_attn = MultiHeadAttn(d_model, num_heads=8)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = LayerNorm(d_model)
        self.multi_head_src_attn = MultiHeadAttn(d_model, num_heads=8)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.norm3 = LayerNorm(d_model)

    def forward(self, src_encoded, x, src_mask, tgt_mask):
        attn_x = self.multi_head_masked_attn(query=x, key=x, value=x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(attn_x))
        attn_x = self.multi_head_src_attn(query=x, key=src_encoded, value=src_encoded, mask=src_mask)
        x = self.norm2(x + self.dropout2(attn_x))
        return self.norm3(x + self.dropout3(self.feed_forward(x)))