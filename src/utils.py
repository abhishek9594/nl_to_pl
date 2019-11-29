#!/usr/bin/env python
from __future__ import division

import math, copy, numpy as np
import torch
import torch.nn as nn
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

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

def read_corpus(file_path, domain):
    """
    read the file delineated line by line and extract tokens using NLTK
    @param file_path (str): path to file containing corpus
    @param domain (str): language domain:  src (source) or tgt (target)
    @return data (list[list[str]]): list of tokens
    """
    data = []
    for line in open(file_path, 'r'):
        toks = nltk.word_tokenize(line)
        if domain == 'tgt':
            toks = ['<start>'] + toks + ['<eos>']
        data.append(toks)

    return data

def subsequent_mask(size):
    mask_shape = (1, size, size)
    subseq_mask = np.triu(np.ones(mask_shape), k=1).astype('uint8')
    return torch.from_numpy(subseq_mask) == 0 #switch 1's & 0's

def clone(block, n):
    return nn.ModuleList([copy.deepcopy(block) for _ in range(n)])

def save_sents(sents, file_path):
    """
    save sentences in a file line by line
    @param sents (list[list[str]]): list of sentences
    @param file_path (str): location to save senetnces
    """
    with open(file_path, 'w') as file_obj:
        for sent in sents:
            file_obj.write(' '.join(sent) + '\n')

def compute_bleu_score(refs, hyps):
    """
    compute bleu score for the given references against hypotheses
    @param refs (list[list[str]]): list of reference sents with <start> and <eos>
    @param hyps (list[list[str]]): list of generated candidate sents
    @return bleu_score (float): BLEU score for the word overlap
    """
    assert len(refs) == len(hyps)
    bleu_score = corpus_bleu([[ref[1:-1]] for ref in refs],
                            hyps,
                            smoothing_function=SmoothingFunction().method4)
    return bleu_score

def compute_exact_match(refs, hyps):
    """
    compute exact match for the given references against hypotheses
    @param refs (list[list[str]]): list of reference sents with <start> and <eos>
    @param hyps (list[list[str]]): list of generated candidate sents
    @return em_score (float): exact match score
    """
    assert len(refs) == len(hyps)
    em = 0
    for ref, hyp in zip(refs, hyps):
        if ref[1:-1] == hyp: em += 1
    return em / len(refs)

def batch_iter(data, batch_size, shuffle=False):
    """
    Yield batches of source and target sentences reverse sorted by source sentences' lengths (largest to smallest)
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentences
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = int(math.ceil(len(data) / batch_size))
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
