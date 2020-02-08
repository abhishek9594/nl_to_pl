#!/usr/bin/env python
from __future__ import division

import math
import numpy as np
import copy
import torch
import torch.nn as nn

def pad_sents(sents, pad_id=0):
    """
    pad the list of sents according to max sent len
    @param sents (list[list[int]]): list of word ids of sentences
    @param pad_id (int): pad idx
    @return sents_padded (list[list[int]]): padded sentences
    """
    sents_padded = []
    max_len = 0

    for sent in sents:
        if len(sent) > max_len: max_len = len(sent)
    for sent in sents:
        sent_padded = sent
        sent_padded.extend([pad_id for i in range(max_len - len(sent))])
        sents_padded.append(sent_padded)

    return sents_padded

def map_src_tgt(src, tgt, vocab, device):
    """
    map src words into tgt sent
    """
    tgt_copy, src_ids = [], []
    max_unk_src_words = 0
    for src_sent, tgt_sent in zip(src, tgt):
        unk_word_idx = map_src_words_tgt(src_sent, vocab)
        tgt_copy.append([vocab.tgt[word] if word not in unk_word_idx else len(vocab.tgt) + unk_word_idx[word] for word in tgt_sent])
        src_ids.append([vocab.tgt[word] if word not in unk_word_idx else len(vocab.tgt) + unk_word_idx[word] for word in src_sent])
        if len(unk_word_idx):
            max_unk_src_words = max(max_unk_src_words, len(unk_word_idx))
    tgt_copy_padded = pad_sents(tgt_copy, vocab.tgt['<pad>'])
    src_ids_padded = pad_sents(src_ids)
    max_tgt_len = max(len(tgt_sent) for tgt_sent in tgt)
    src_ids_expand = [[src_ids_padded[i]] * max_tgt_len for i in range(len(src_ids_padded))]
    return torch.tensor(tgt_copy_padded, dtype=torch.long, device=device), torch.tensor(src_ids_expand, dtype=torch.long, device=device), max_unk_src_words

def map_src_words_tgt(src_sent, vocab):
    unk_word_idx = dict()
    unk_word_pos = 0
    for word in src_sent:
        if word not in vocab.tgt and word not in unk_word_idx:
            unk_word_idx[word] = unk_word_pos
            unk_word_pos += 1
    return unk_word_idx

def subsequent_mask(size):
    mask_shape = (1, size, size)
    subseq_mask = np.triu(np.ones(mask_shape), k=1).astype('uint8')
    return torch.from_numpy(subseq_mask) == 0 #switch 1's & 0's

def clone(block, n):
    return nn.ModuleList([copy.deepcopy(block) for _ in range(n)])
