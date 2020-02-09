#!/usr/bin/env python

"""
model for tuning tokens' embeddings
"""

import torch.nn as nn

class ModelEmbeddings(nn.Module):

    def __init__(self, embed_size, vocab, node_map):
        """
        @param embed_size (int): embedding size
        @param vocab (Vocab): obj containing src and tgt VocabEntry
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size

        src_pad_idx = vocab.src['<pad>']
        tgt_pad_idx = vocab.tgt['<pad>']
        node_pad_idx = node_map['<pad>']
        
        self.src_embedding = nn.Embedding(len(vocab.src), self.embed_size, padding_idx=src_pad_idx)
        self.gen_tok_embedding = nn.Embedding(len(vocab.tgt), self.embed_size, padding_idx=tgt_pad_idx)
        self.node_embedding = nn.Embedding(len(node_map.keys()), self.embed_size, padding_idx=node_pad_idx)