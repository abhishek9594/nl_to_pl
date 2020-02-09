#!/usr/bin/env python
from __future__ import division

import math
import numpy as np
import pickle
import copy
import torch
import torch.nn as nn

NODE_MAP = {
    '<pad>' = 0,
    'root' = 1,
    'ImportFrom' = 2,
    'alias*' = 3,
    'alias' = 4,
    'Import' = 5,
    'Assign' = 6,
    'expr*' = 7,
    'expr' = 8,
    'Name' = 9,
    'List' = 10,
    'If' = 11,
    'Compare' = 12,
    'cmpop*' = 13,
    'cmpop' = 14,
    'Attribute' = 15,
    'stmt*' = 16,
    'stmt' = 17,
    'Raise' = 18,
    'Call' = 19,
    'BinOp' = 20,
    'operator' = 21,
    'FunctionDef' = 22,
    'arguments' = 23,
    'Expr' = 24,
    'keyword*' = 25,
    'keyword' = 26,
    'Num' = 27,
    'Return' = 28,
    'TryExcept' = 29,
    'excepthandler*' = 30,
    'ExceptHandler' = 31,
    'Subscript' = 32,
    'slice' = 33,
    'Index' = 34,
    'Tuple' = 35,
    'Str' = 36,
    'ClassDef' = 37,
    'Dict' = 38,
    'For' = 39,
    'IfExp' = 40,
    'UnaryOp' = 41,
    'unaryop' = 42,
    'BoolOp' = 43,
    'boolop' = 44,
    'With' = 45,
    'TryFinally' = 46,
    'ListComp' = 47,
    'comprehension*' = 48,
    'comprehension' = 49,
    'Delete' = 50,
    'AugAssign' = 51,
    'Lambda' = 52,
    'GeneratorExp' = 53,
    'Assert' = 54,
    'Yield' = 55,
    'While' = 56,
    'Slice' = 57,
    'DictComp' = 58,
    'Print' = 59,
    'Exec' = 60,
    'Global' = 61,
    'str*' = 62
}

RULE_MAP = pickle.load(open('rule.pickle', 'rb'))

def wrapGenTok(word):
    #wrap word inside `GenToken[]`
    return 'GenToken[' + word + ']'

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

def src_tensor(src_sents, src_vocab, device):
    word_ids = src_vocab.words2indices(sents)
    sents_padded = pad_sents(word_ids, src_vocab['<pad>'])
    sents_tensor = torch.tensor(sents_padded, dtype=torch.long, device=device)
    return sents_tensor

def tgt_tensors(tgt_sents, tgt_vocab, device):
    gen_tok_ids = [[tgt_vocab[tok] if 'GenToken' in tok else tgt_vocab['<pad>'] for tok in sent] for sent in tgt_sents]
    node_ids = [[tgt_vocab['<pad>'] if 'GenToken' in tok else NODE_MAP[tok[: tok.find('->') - 1]] for tok in sent] for sent in tgt_sents]
    gen_toks_padded = pad_sents(gen_tok_ids, tgt_vocab['<pad>'])
    nodes_padded = pad_sents(node_ids, NODE_MAP['<pad>'])
    return torch.tensor(gen_toks_padded, dtype=torch.long, device=device), torch.tensor(nodes_padded, dtype=torch.long, device=device)

def map_src_tgt(src, tgt, vocab, device):
    """
    map src words into tgt sent, and src words into tgt vocab
    @return tgt_copy: tensor which accounts src_word for <unk> tokens (b, Q)
    @return: src_ids_map_tgt: src_id -> TargetVocab(src_id) (b,Q,K)
    """
    tgt_copy, src_ids_map_tgt = [], []
    max_unk_src_words = 0
    for src_sent, tgt_sent in zip(src, tgt):
        unk_word_idx = map_src_words_tgt(src_sent, vocab)
        tgt_copy.append([vocab.tgt[word] if word not in unk_word_idx else len(vocab.tgt) + unk_word_idx[word] for word in tgt_sent])
        src_ids_map_tgt.append([vocab.tgt[wrapGenTok(word)] if wrapGenTok(word) not in unk_word_idx else len(vocab.tgt) + unk_word_idx[wrapGenTok(word)] for word in src_sent])
        if len(unk_word_idx):
            max_unk_src_words = max(max_unk_src_words, len(unk_word_idx))
    tgt_copy_padded = pad_sents(tgt_copy, vocab.tgt['<pad>'])
    src_ids_map_tgt_padded = pad_sents(src_ids_map_tgt, vocab.src['<pad>'])
    max_tgt_len = max(len(tgt_sent) for tgt_sent in tgt)
    #expand to adjust for Q length tgt_sent in (b,Q,K), as any word in a query (Q[i]) can come from src_sent
    src_ids_map_tgt_expand = [[src_ids_map_tgt_padded[i]] * max_tgt_len for i in range(len(src_ids_map_tgt_padded))]
    return torch.tensor(tgt_copy_padded, dtype=torch.long, device=device), torch.tensor(src_ids_map_tgt_expand, dtype=torch.long, device=device), max_unk_src_words

def map_src_words_tgt(src_sent, vocab):
    unk_word_idx = dict()
    unk_word_pos = 0
    for word in src_sent:
        word = wrapGenTok(word)
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