#!/usr/bin/env python

"""
model for tuning tokens' embeddings
"""

import torch.nn as nn

class ModelEmbeddings(nn.Module):

    def __init__(self, embed_size, vocab, nodes):
        """
        @param embed_size (int): embedding size
        @param vocab (Vocab): Vocab obj containing src and tgt VocabEntry
        @param nodes (Node): Node obj containing node mapping to id
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size

        self.src_embedding = nn.Embedding(len(vocab.src), self.embed_size, padding_idx=vocab.src.pad_id)
        self.tgt_node_embedding = nn.Embedding(len(nodes), self.embed_size, padding_idx=nodes.pad_id)
        self.tgt_token_embedding = nn.Embedding(len(vocab.tgt), self.embed_size, padding_idx=vocab.tgt.pad_id)
