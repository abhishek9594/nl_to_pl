#!/usr/bin/env python
from __future__ import division

import math
import torch, torch.nn as nn
from torch.autograd import Variable

class PositionalEmbeddings(nn.Module):

    def __init__(self, d_model, dropout_rate, max_seq_len=1024):
        super(PositionalEmbeddings, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** (2 * (i + 1) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.shape[1]], requires_grad=False)
        return self.dropout(x)
