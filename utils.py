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
