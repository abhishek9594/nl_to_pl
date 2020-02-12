#!/usr/bin/env python
"""
node.py: Map all the node types of the PL grammar to node_id

Usage:
    node.py --train-src=<file> --train-tgt=<file> NODE_FILE [options]

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
from lang.py.grammar import PY_AST_NODE_FIELDS

class Node(object):
    def __init__(self, node2id=None):
        """
        @param node2id (dict): dictionary mapping nodes -> indices
        """
        if node2id:
            self.node2id = node2id
        else:
            self.node2id = dict()
            self.node2id['<pad>'] = 0       #Pad Token
        self.pad_id = self.node2id['<pad>']
        self.id2node = {v: k for k, v in self.node2id.items()}

    def __getitem__(self, node):
        """ Retrieve node's index.
        @param node (str): node to look up.
        @returns index (int): index of the node 
        """
        return self.node2id.get(node)

    def __contains__(self, node):
        """ Check if node is captured by Node.
        @param node (str): node to look up
        @returns contains (bool): whether node is contained    
        """
        return node in self.node2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the Node.
        """
        raise ValueError('Node dictionary is readonly')

    def __len__(self):
        """ Compute number of nodes in Node.
        @returns len (int): number of nodes in Node
        """
        return len(self.node2id)

    def __repr__(self):
        """ Representation of Node to be used
        when printing the object.
        """
        return 'Node[size=%d]' % len(self)

    def id2node(self, n_id):
        """ Return mapping of index to node.
        @param n_id (int): node index
        @returns node (str): node corresponding to index
        """
        return self.id2node[n_id]

    def add(self, node):
        """ Add node to Node, if it is previously unseen.
        @param node (str): node to add to Node
        @return index (int): index that the node has been assigned
        """
        if node not in self:
            n_id = self.node2id[node] = len(self)
            self.id2node[n_id] = node
            return n_id
        else:
            return self[node]

    def nodes2indices(self, sents):
        """ Convert list of tokens or list of sentences of tokens
        into list or list of list of indices.
        @param sents (list[str] or list[list[str]]): sentence(s) containing either node or GenToken toks
        @return node_ids (list[int] or list[list[int]]): sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [[self[node] if 'GenToken' not in node else self.pad_id for node in sent] for sent in sents]
        else:
            sent = sents
            return [self[node] if 'GenToken' not in node else self.pad_id for node in sent]

    def indices2nodes(self, node_ids):
        """ Convert list of indices into nodes.
        @param node_ids (list[int]): list of node ids
        @return sents (list[str]): list of nodes
        """
        return [self.id2node[n_id] for n_id in node_ids]

    def sents2Tensor(self, sents):
        """
        Convert list of tgt sents to node tensor by padding required sents
        where tgt sents can contain node and GenToken toks
        @param sents (list[list[str]]): batch of tgt sents
        @return node_tensor (torch.tensor (max_sent_len, batch_size))
        """
        node_ids = self.nodes2indices(sents)
        nodes_padded = pad_sents(node_ids, self.pad_id)
        return torch.tensor(nodes_padded, dtype=torch.long)

    @staticmethod
    def build(corpus):
        """ Given a corpus of list of list of nodes construct a Node obj.
        @param corpus (list[list[str]]): corpus of data_nodes constructed in the main
        @returns nodes (Node): Node instance produced from provided corpus
        """
        nodes = Node()
        node_freq = Counter(chain(*corpus))
        top_nodes = sorted(node_freq.keys(), key=lambda n: node_freq[n], reverse=True)
        for node in top_nodes:
            nodes.add(node)
        for node in PY_AST_NODE_FIELDS:
            if node not in nodes: nodes.add(node)
        return nodes

    def save(self, file_path):
        """ Save Node to file as pickle dump.
        @param file_path (str): file path to node file
        """
        pickle.dump(self.node2id, open(file_path, 'wb'))

    @staticmethod
    def load(file_path):
        """
        @param file_path (str): file path to node file
        @returns Node object loaded from pickle dump
        """
        node2id = pickle.load(open(file_path, 'rb'))

        return Node(node2id)

if __name__ == '__main__':
    args = docopt(__doc__)

    print('read in source sentences: %s' % args['--train-src'])
    print('read in target sentences: %s' % args['--train-tgt'])

    (_, tgt_sents), _ = read_corpus(args['--train-src'], args['--train-tgt'])
    tgt_tokens = [parse(code).to_tokens() for code in tgt_sents]
    #filter GenTokens
    data_nodes = [[token for token in tokens if 'GenToken' not in token] for tokens in tgt_tokens]

    nodes = Node.build(data_nodes)
    print('generated nodes: %d' % (len(nodes)))

    nodes.save(args['NODE_FILE'])
    print('nodes saved to %s' % args['NODE_FILE'])