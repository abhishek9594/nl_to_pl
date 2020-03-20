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

def batch_iter(src_sents, tgt_sents, lang, batch_size, shuffle=False):
    """
    Yield batches of source and target sentences reverse sorted by source sentences' lengths (largest to smallest)
    @param src_sents (list(list[str])): list of source sentences (list of tokens)
    @param tgt_sents (list[str]): list of target sentences
    @param lang: target language
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    if lang == 'lambda':
        from lang.Lambda.lambda_transition_system import LambdaCalculusTransitionSystem
        from lang.Lambda.transition_system import ApplyRuleAction, ReduceAction, GenTokenAction
        from lang.Lambda.asdl import ASDLGrammar
        from lang.Lambda.parse import parse_lambda_expr, logical_form_to_ast

        asdl_desc = open('lang/Lambda/lambda_asdl.txt').read()
        grammar = ASDLGrammar.from_text(asdl_desc)
        parser = LambdaCalculusTransitionSystem(grammar)

        tgt_asts = [logical_form_to_ast(grammar, parse_lambda_expr(sent)) for sent in tgt_sents]
        tgt_actions = [parser.get_actions(ast) for ast in tgt_asts]

        tgt_nodes, tgt_rules = [], []
        for actions in tgt_actions:
            tgt_node, tgt_rule = [], []
            nodes = ['<start>']
            for action in actions:
                assert len(nodes) > 0
                if isinstance(action, ApplyRuleAction):
                    if nodes[-1][-1] == '*':
                        tgt_node.append(nodes[-1])
                    else:
                        tgt_node.append(nodes.pop())

                    rule = action.production #ASDLProduction(ASDLType(type), ASDLConstructor(constructor))
                    fields = rule.constructor.fields #fields[Field(name, type, cardinality)]
                    curr_nodes = []
                    for field in fields: #Field(name, ASDLType(name), cardinality)
                        node_name = field.type.name
                        field_cardinality = field.cardinality
                        if field_cardinality == 'multiple':
                            node_name += '*'
                        curr_nodes.append(node_name)
                    nodes.extend(curr_nodes[::-1])

                    tgt_rule.append(rule)
                elif isinstance(action, GenTokenAction):
                    tgt_node.append(nodes.pop())
                    tgt_rule.append(action.token)
                else:
                    tgt_node.append(nodes.pop())
                    tgt_rule.append('Reduce')
            tgt_nodes.append(tgt_node)
            tgt_rules.append(tgt_rule)

        data = zip(src_sents, tgt_nodes, tgt_rules)

        batch_num = int(math.ceil(len(data) / batch_size))
        index_array = list(range(len(data)))

        if shuffle:
            np.random.shuffle(index_array)

        for i in range(batch_num):
            indices = index_array[i * batch_size: (i + 1) * batch_size]
            examples = [data[idx] for idx in indices]

            examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
            src_sents = [e[0] for e in examples]
            tgt_nodes = [e[1] for e in examples]
            tgt_rules = [e[2] for e in examples]

            yield src_sents, tgt_nodes, tgt_sents
    else:
        print('language:  %s currently not supported' % (lang))