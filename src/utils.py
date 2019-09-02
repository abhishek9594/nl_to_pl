#!/usr/bin/env python
from __future__ import division

import numpy as np
import math
import nltk

def pad_sents(sents, pad_id):
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
