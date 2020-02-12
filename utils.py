#!/usr/bin/env python
from __future__ import division

import math
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import re

QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?<!\\)(?P=quote)")

def wrapGenTok(tok):
    #wrap tok inside `GenToken[]`
    return 'GenToken[' + tok + ']'

def read_corpus(src_file, tgt_file):
    """
    extract input tokens using NLTK, whereas parse output tokens according to its syntax
    @param file_path (str): path to file containing corpus
    @return data ((list[list[str]], list[list[str]])): tuples of list of src and tgt tokens
    @return sent_str_map {sent_num -> str_map{str -> str_repr}}: mapping sent_id to str_map (maps str_repr to str), useful for mapping quoted strings
    """
    src_data, tgt_data = [], []
    sent_str_map = dict()
    for sent_num, src_sent in enumerate(open(src_file, 'r')):
        str_map = dict()
        str_set = set()
        str_count = 0
        matches = QUOTED_STRING_RE.findall(src_sent)
        for quote, raw_str in matches:
            if raw_str in str_set:
                continue
            str_repr = '_STR:%d_' % str_count
            str_literal = quote + raw_str + quote
            str_map[str_repr] = str_literal
            str_set.add(raw_str)

            src_sent = src_sent.replace(str_literal, str_repr)
            str_count += 1

        src_toks = nltk.word_tokenize(src_sent)
        src_data.append(src_toks)
        if len(str_map) > 0: sent_str_map[sent_num] = str_map

    for sent_num, tgt_sent in enumerate(open(tgt_file, 'r')):
        if sent_num in sent_str_map:
            for str_repr, str_literal in sent_str_map[sent_num].items():
                tgt_sent = tgt_sent.replace(str_literal, str_repr)
        tgt_data.append(tgt_sent.rstrip())
        
    data = zip(src_data, tgt_data)
    return (src_data, tgt_data), sent_str_map

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
    bleu_score = corpus_bleu([[ref[1:-1]] for ref in refs],
                            hyps,
                            smoothing_function=SmoothingFunction().method4)
    return bleu_score

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
