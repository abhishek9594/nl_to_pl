#!/usr/bin/env python
from __future__ import division

import math
import numpy as np
import nltk

def wrapGenTok(tok):
    #wrap tok inside `GenToken[]`
    return 'GenToken[' + tok + ']'

def read_corpus(data_file):
    """
    extract input tokens using NLTK, whereas leave output tokens unformatted
    @param data_file (str): file containing src_sent and tgt_sent sep by '\t', each line is delineated by '\n'
    @return data ((list[list[str]], list[str])): tuples of list of src and tgt tokens
    """
    src_data, tgt_data = [], []
    for sent in open(data_file, 'r'):
        sents = sent.split('\n')[0].split('\t') #[src_sent, tgt_sent]
        src_sent = sents[0]
        tgt_sent = sents[1]
        
        src_sent_toks = nltk.word_tokenize(src_sent)
        src_data.append(src_sent_toks)

        tgt_data.append(tgt_sent)

    return src_data, tgt_data

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

def save_sents(sents, file_path):
    """
    save sentences in a file line by line
    @param sents (list[str]): list of sentences
    @param file_path (str): location to save senetnces
    """
    with open(file_path, 'w') as file_obj:
        for sent in sents:
            file_obj.write(sent + '\n')

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