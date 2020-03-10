#!/usr/bin/env python
"""
vocab.py: Vocabulary Generation

Usage:
    vocab.py --lang=<str> --data=<file> VOCAB_FILE [options]

Options:
    -h --help                   Show this screen.
    --lang=<str>                target language
    --data=<file>               file containing src and tgt sents
    --freq-cutoff=<int>         frequency cutoff [default: 1]
"""

from collections import Counter
from docopt import docopt
from itertools import chain
import pickle
import torch

from utils import read_corpus, pad_sents, wrapGenTok


class VocabEntry(object):
    def __init__(self, word2id=None):
        """
        @param word2id (dict): dictionary mapping words -> indices
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0       #pad Token
            self.word2id['<start>'] = 1     #start Token
            self.word2id['<eos>'] = 2       #end Token
            self.word2id['<unk>'] = 3       #unknown Token
        self.pad_id = self.word2id['<pad>']
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word 
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """ Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained    
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        """ Return mapping of index to word.
        @param wid (int): word index
        @returns word (str): word corresponding to index
        """
        return self.id2word[wid]

    def add(self, word):
        """ Add word to VocabEntry, if it is previously unseen.
        @param word (str): word to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words
        into list or list of list of indices.
        @param sents (list[str] or list[list[str]]): sentence(s) in words
        @return word_ids (list[int] or list[list[int]]): sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        @param word_ids (list[int]): list of word ids
        @return sents (list[str]): list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def sents2Tensor(self, sents):
        """
        Convert list of sent to tensor by padding required sents
        @param sents (list[list[str]]): batch of sents in reverse sorted order
        @return out_tensor (torch.tensor (max_sent_len, batch_size))
        """
        word_ids = self.words2indices(sents)
        sents_padded = pad_sents(word_ids, self['<pad>'])
        return torch.tensor(sents_padded, dtype=torch.long) #(b, Q)

    @staticmethod
    def from_corpus(corpus, freq_cutoff):
        """ Given a corpus construct a Vocab Entry.
        @param corpus (list[list[str]]): corpus of text produced by read_corpus function
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
        @returns vocab_entry (VocabEntry): VocabEntry instance produced from provided corpus
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry

class Vocab(object):
    """ 
    Vocab encapsulating src and target langauges.
    """
    def __init__(self, src_vocab, tgt_vocab):
        """
        @param src_vocab (VocabEntry): VocabEntry for source language
        @param tgt_vocab (VocabEntry): VocabEntry for target language
        """
        self.src = src_vocab
        self.tgt = tgt_vocab

    def tgt_sents2Tensor(self, tgt_sents):
        """
        Convert list of tgt_sents to tensor by padding required sents
        where a tgt_sent could contain nodes + GenTokens or rules + GenTokens
        @param sents (list[list[str]]): batch of tgt_sents
        @return out_tensor (torch.tensor (max_sent_len, batch_size))
        """
        #filter any non-GenToken
        tgt_tokens = [[tok if 'GenToken' in tok else '<pad>' for tok in sent] for sent in tgt_sents]
        return self.tgt.sents2Tensor(tgt_tokens)
    
    def copy_gen_tok_ids(self, src_sents, tgt_sents):
        """
        get tgt_gen_tok ids tensor representation for tgt_sents
        where we account src_toks copied to tgt_sents
        @param src_sents (list[list[str]]): list of source sentences
        @param tgt_sents (list[list[str]]): list of target sentences
        @return tgt_gen_toks_tensor (torch.tensor (max_tgt_sent_len, batch_size))
        """
        tgt_gen_toks = []
        for src_sent, tgt_sent in zip(src_sents, tgt_sents):
            unk_word_idx = self.map_unk_src(src_sent)
            tgt_gen_toks.append([self.tgt['<pad>'] if 'GenToken' not in tok else self.tgt[tok] if tok not in unk_word_idx else len(self.tgt) + unk_word_idx[tok] for tok in tgt_sent])
        tgt_gen_toks_padded = pad_sents(tgt_gen_toks, self.tgt['<pad>'])
        return torch.tensor(tgt_gen_toks_padded, dtype=torch.long) #(b, Q)

    def src_ids_in_tgt(self, src_sents, tgt_sents):
        """
        compute src_tok id in vocab.tgt
        idx will be used while copying src_tok to tgt_sent
        @param src_sents (list[list[str]]): list of source sentences
        @param tgt_sents (list[list[str]]): list of target sentences
        @return src_ids_map_tgt_tensor (torch.tensor (max_src_sent_len, batch_size))
        """
        src_ids_map_tgt = []
        for src_sent in src_sents:
            unk_word_idx = self.map_unk_src(src_sent)
            src_ids_map_tgt.append([self.tgt[wrapGenTok(tok)] if wrapGenTok(tok) not in unk_word_idx else len(self.tgt) + unk_word_idx[wrapGenTok(tok)] for tok in src_sent])
        src_ids_map_tgt_padded = pad_sents(src_ids_map_tgt, self.src['<pad>'])
        max_tgt_len = max(len(tgt_sent) for tgt_sent in tgt_sents)
        #expand to adjust for Q length tgt_sent in (b,Q,K), as any word in a query (Q[i]) can come from src_sent
        src_ids_map_tgt_expanded = [[src_ids_map_tgt_padded[i]] * max_tgt_len for i in range(len(src_ids_map_tgt))]
        return torch.tensor(src_ids_map_tgt_expanded, dtype=torch.long) #(b,Q,K)

    def max_unk_toks(self, src_sents):
        """
        compute max unk_toks for vocab.tgt occuring in src_sents
        @param src_sents (list[list[str]]): list of source sentences
        @return max_src_unk_toks (int): max src_unk toks for vocab.tgt
        """
        return max([len(self.map_unk_src(src_sent)) for src_sent in src_sents])

    def map_unk_src(self, src_sent):
        """
        map: src_tok -> id, such that self.tgt(src_tok) = unk_id
        map all the sorce tokens which are <unk> in target vocab to ids
        @param src_sent (list[str]): src sent containing list of tokens
        @return unk_word_idx (dict(unk_word : id)): dictionary mapping the unk src tokens to id
        """
        unk_word_idx = dict()
        unk_word_pos = 0
        for word in src_sent:
            word = wrapGenTok(word)
            if word not in self.tgt and word not in unk_word_idx:
                unk_word_idx[word] = unk_word_pos
                unk_word_pos += 1
        return unk_word_idx

    @staticmethod
    def build(src_sents, tgt_sents, freq_cutoff):
        """
        @param src_sents (list[list[str]]): Source sentences provided by read_corpus() function
        @param tgt_sents (list[list[str]]): Target sentences provided by read_corpus() function
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word.
        """
        assert len(src_sents) == len(tgt_sents)

        src = VocabEntry.from_corpus(src_sents, freq_cutoff)

        tgt = VocabEntry.from_corpus(tgt_sents, freq_cutoff)

        return Vocab(src, tgt)

    def save(self, file_path):
        """ Save Vocab to file as pickle dump.
        @param file_path (str): file path to vocab file
        """
        pickle.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), open(file_path, 'wb'))

    @staticmethod
    def load(file_path):
        """
        @param file_path (str): file path to vocab file
        @returns Vocab object loaded from pickle dump
        """
        entry = pickle.load(open(file_path, 'rb'))
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']

        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))

if __name__ == '__main__':
    args = docopt(__doc__)

    print('read in sentences: %s' % args['--data'])

    src_sents, tgt_sents = read_corpus(args['--data'])

    lang = args['--lang']
    if lang == 'lambda':
        from lang.Lambda.lambda_transition_system import LambdaCalculusTransitionSystem
        from lang.Lambda.transition_system import GenTokenAction
        from lang.Lambda.asdl import ASDLGrammar
        from lang.Lambda.parse import parse_lambda_expr, logical_form_to_ast

        asdl_desc = open('lang/Lambda/lambda_asdl.txt').read()
        grammar = ASDLGrammar.from_text(asdl_desc)
        parser = LambdaCalculusTransitionSystem(grammar)
        
        tgt_asts = [logical_form_to_ast(grammar, parse_lambda_expr(sent)) for sent in tgt_sents]
        tgt_actions = [parser.get_actions(ast) for ast in tgt_asts]
        tgt_tokens = [[action.token for action in actions if isinstance(action, GenTokenAction)] for actions in tgt_actions]

        vocab = Vocab.build(src_sents, tgt_tokens, int(args['--freq-cutoff']))
        print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.tgt)))

        vocab.save(args['VOCAB_FILE'])
        print('vocabulary saved to %s' % args['VOCAB_FILE'])
    else:
        print('language:  %s currently not supported' % (lang))
