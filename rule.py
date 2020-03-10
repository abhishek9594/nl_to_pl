#!/usr/bin/env python
"""
rule.py: Map all the rule types of the PL grammar to rule_id

Usage:
    rule.py --lang=<str> RULE_FILE [options]

Options:
    -h --help                   Show this screen.
    --lang=<str>                target language
"""

from docopt import docopt
import pickle
import torch

class Rule(object):
    def __init__(self, rule2id=None):
        """
        @param rule2id (dict): dictionary mapping rules -> indices
        """
        if rule2id:
            self.rule2id = rule2id
        else:
            self.rule2id = dict()
            self.rule2id['<pad>'] = 0       #Pad token
            self.rule2id['Reduce'] = 1      #Reduce action token
        self.pad_id = self.rule2id['<pad>']
        self.reduce_id = self.rule2id['Reduce']
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

    def sents2Tensor(self, sents):
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
    def build(grammar):
        """ Given a grammar (ASDL) description of language, extract all production rules
        @param grammar (ASDLGrammar): grammar object described in the asdl file for the target language
        @returns rules (Rule): Rule instance produced from the grammar
        """
        rules = Rule()
        for production in grammar.productions:
            rules.add(production)
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

    lang = args['--lang']
    if lang == 'lambda':
        from lang.Lambda.asdl import ASDLGrammar

        asdl_desc = open('lang/Lambda/lambda_asdl.txt').read()
        grammar = ASDLGrammar.from_text(asdl_desc)

        rules = Rule.build(grammar)
        print('generated rules: %d' % (len(rules)))

        rules.save(args['RULE_FILE'])
        print('rules saved to %s' % args['RULE_FILE'])
    else:
        print('language:  %s currently not supported' % (lang))