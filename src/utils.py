#!/usr/bin/env python

import nltk

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
    batch_num = math.ceil(len(data) / batch_size)
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
