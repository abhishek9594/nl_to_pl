#!/usr/bin/env python
from __future__ import division

import numpy as np
import copy
import torch
import torch.nn as nn

def subsequent_mask(size):
    mask_shape = (1, size, size)
    subseq_mask = np.triu(np.ones(mask_shape), k=1).astype('uint8')
    return torch.from_numpy(subseq_mask) == 0 #switch 1's & 0's

def clone(block, n):
    return nn.ModuleList([copy.deepcopy(block) for _ in range(n)])