#!/usr/bin/env python

"""
model for tuning tokens' embeddings
"""

import torch.nn as nn

class ModelEmbeddings(nn.Module):

    def __init__(self, embed_size, vocab):
        """
        @param embed_size (int): embedding size
        @param vocab (Vocab): obj containing src and tgt VocabEntry
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size

        src_pad_idx = vocab.src['<pad>']
        tgt_pad_idx = vocab.src['<pad>']
        
        self.src_embedding = nn.Embedding(len(vocab.src), self.embed_size, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(len(vocab.tgt), self.embed_size, padding_idx=tgt_pad_idx)