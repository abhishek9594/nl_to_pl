#/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from utils import clone

class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super(MultiHeadAttn, self).__init__()
        self.num_heads = num_heads
        d_k = d_v = d_model // self.num_heads
        self.query_projects = clone(nn.Linear(d_model, d_k, bias=False), n=num_heads)
        self.key_projects = clone(nn.Linear(d_model, d_k, bias=False), n=num_heads)
        self.value_projects = clone(nn.Linear(d_model, d_v, bias=False), n=num_heads)
        self.out_project = nn.Linear(self.num_heads * d_v, d_model, bias=False)

    def forward(self, query, key, value, mask):
        attn_outs = []
        q_key_dots = []
        for h in range(self.num_heads):
            query_mapped = self.query_projects[h](query)
            key_mapped = self.key_projects[h](key)
            value_mapped = self.value_projects[h](value)
            attn_out, q_key_dot = self.attn(query_mapped, key_mapped, value_mapped, mask)
            attn_outs.append(attn_out)
            q_key_dots.append(q_key_dot)
        out = torch.cat([attn_out for attn_out in attn_outs], dim=-1).to(query.device)
        return self.out_project(out), q_key_dots

    def attn(self, query, key, value, mask):
        d_k = key.shape[-1]
        assert d_k == query.shape[-1]
        scores = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(d_k) #(b, Q, K)
        if mask is not None: scores = scores.masked_fill(mask == 0, -float('inf'))
        p_attn = F.softmax(scores, dim=-1)
        return torch.bmm(p_attn, value), scores #(b, Q, d_v)
