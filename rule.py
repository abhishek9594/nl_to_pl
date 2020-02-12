#!/usr/bin/env python
"""
rule.py: Map all the rule types of the PL grammar to rule_id

Usage:
    rule.py --train-src=<file> --train-tgt=<file> RULE_FILE [options]

Options:
    -h --help                  Show this screen.
    --train-code=<file>        File containing all the code
"""

from collections import Counter
from docopt import docopt
from itertools import chain
import pickle
import torch

from utils import read_corpus, pad_sents
from parse import parse

class Rule(object):
    def __init__(self, rule2id=None):
        """
        @param rule2id (dict): dictionary mapping rules -> indices
        """
        if rule2id:
            self.rule2id = rule2id
        else:
            self.rule2id = dict()
            self.rule2id['<pad>'] = 0       #Pad Token
        self.pad_id = self.rule2id['<pad>']
        self.id2rule = {v: k for k, v in self.rule2id.items()}

    def __getitem__(self, rule):
        """ Retrieve rule's index.
        @param rule (str): rule to look up.
        @returns index (int): index of the rule 
        """
        return self.rule2id.get(rule)

    def __contains__(self, rule):
        """ Check if rule is captured by Rule.
        @param rule (str): rule to look up
        @returns contains (bool): whether rule is contained    
        """
        return rule in self.rule2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the Rule.
        """
        raise ValueError('Rule dictionary is readonly')

    def __len__(self):
        """ Compute number of rules in Rule.
        @returns len (int): number of rules in Rule
        """
        return len(self.rule2id)

    def __repr__(self):
        """ Representation of Rule to be used
        when printing the object.
        """
        return 'Rule[size=%d]' % len(self)

    def id2rule(self, n_id):
        """ Return mapping of index to rule.
        @param n_id (int): rule index
        @returns rule (str): rule corresponding to index
        """
        return self.id2rule[n_id]

    def add(self, rule):
        """ Add rule to Rule, if it is previously unseen.
        @param rule (str): rule to add to Rule
        @return index (int): index that the rule has been assigned
        """
        if rule not in self:
            n_id = self.rule2id[rule] = len(self)
            self.id2rule[n_id] = rule
            return n_id
        else:
            return self[rule]

    def rules2indices(self, sents):
        """ Convert list of tokens or list of sentences of tokens
        into list or list of list of indices.
        @param sents (list[str] or list[list[str]]): sentence(s) containing either rule or GenToken toks
        @return rule_ids (list[int] or list[list[int]]): sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [[self[rule] if 'GenToken' not in rule else self.pad_id for rule in sent] for sent in sents]
        else:
            sent = sents
            return [self[rule] if 'GenToken' not in rule else self.pad_id for rule in sent]

    def indices2rules(self, rule_ids):
        """ Convert list of indices into rules.
        @param rule_ids (list[int]): list of rule ids
        @return sents (list[str]): list of rules
        """
        return [self.id2rule[n_id] for n_id in rule_ids]

    def sents2tensor(self, sents):
        """
        Convert list of tgt sents to rule tensor by padding required sents
        where tgt sents can contain rule and GenToken toks
        @param sents (list[list[str]]): batch of tgt sents
        @return rule_tensor (torch.tensor (max_sent_len, batch_size))
        """
        rule_ids = self.rules2indices(sents)
        rules_padded = pad_sents(rule_ids, self.pad_id)
        return torch.tensor(rules_padded, dtype=torch.long)

    @staticmethod
    def build(corpus):
        """ Given a corpus of list of list of rules construct a Rule obj.
        @param corpus (list[list[str]]): corpus of data_rules constructed in the main
        @returns rules (Rule): Rule instance produced from provided corpus
        """
        rules = Rule()
        rule_freq = Counter(chain(*corpus))
        top_rules = sorted(rule_freq.keys(), key=lambda r: rule_freq[r], reverse=True)
        for rule in top_rules:
            rules.add(rule)
        return rules

    def save(self, file_path):
        """ Save Rule to file as pickle dump.
        @param file_path (str): file path to rule file
        """
        pickle.dump(self.rule2id, open(file_path, 'wb'))

    @staticmethod
    def load(file_path):
        """
        @param file_path (str): file path to rule file
        @returns Rule object loaded from pickle dump
        """
        rule2id = pickle.load(open(file_path, 'rb'))

        return Rule(rule2id)

if __name__ == '__main__':
    args = docopt(__doc__)

    print('read in source sentences: %s' % args['--train-src'])
    print('read in target sentences: %s' % args['--train-tgt'])

    (_, tgt_sents), _ = read_corpus(args['--train-src'], args['--train-tgt'])
    tgt_tokens = [parse(code).to_rules() for code in tgt_sents]
    #filter GenTokens
    data_rules = [[token for token in tokens if 'GenToken' not in token] for tokens in tgt_tokens]

    rules = Rule.build(data_rules)
    print('generated rules: %d' % (len(rules)))

    rules.save(args['RULE_FILE'])
    print('rules saved to %s' % args['RULE_FILE'])