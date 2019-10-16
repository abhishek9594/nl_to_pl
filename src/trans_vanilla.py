from __future__ import division
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from trans_encoder import TransEncoder
from trans_decoder import TransDecoder
from model_embeddings import ModelEmbeddings
from utils import subsequent_mask, clone

class TransVanilla(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate):
        super(TransVanilla, self).__init__()
        self.embeddings = ModelEmbeddings(embed_size, vocab)
        self.d_model = embed_size
        self.d_ff = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        
        self.encoder_blocks = clone(TransEncoder(self.d_model, self.d_ff, self.dropout_rate), n=6)
        self.decoder_blocks = clone(TransDecoder(self.d_model, self.d_ff, self.dropout_rate), n=6)
        self.vocab_project = nn.Linear(self.d_model, len(self.vocab.tgt), bias=False)

    def forward(self, src, tgt, padx=0):
        """
        src: (list[list[str]])
        tgt: (list[list[str]])
        """
        src_padded = self.vocab.src.sents2Tensor(src).to(self.device)
        src_mask = (src_padded != padx).unsqueeze(1)
        src_encoded = self.encode(src_padded, src_mask)

        tgt_padded = self.vocab.tgt.sents2Tensor(tgt).to(self.device)
        tgt_input = tgt_padded[:, :-1]
        tgt_mask = (tgt_input != padx).unsqueeze(1)
        subseq_mask = subsequent_mask(tgt_input.shape[-1]).type_as(tgt_mask.data).to(self.device)
        tgt_mask = tgt_mask & subseq_mask
        tgt_decoded = self.decode(src_encoded, tgt_input, src_mask, tgt_mask)
        P = F.softmax(tgt_decoded, dim=-1)
        return P
        
    def encode(self, src, src_mask=None):
        x = self.embeddings.src_embedding(src)
        for encoder in self.encoder_blocks:
            x = encoder(x, src_mask)
        return x

    def decode(self, src_encoded, tgt, src_mask=None, tgt_mask=None):
        x = self.embeddings.tgt_embedding(tgt)
        for decoder in self.decoder_blocks:
            x = decoder(src_encoded, x, src_mask, tgt_mask)
        return self.vocab_project(x)

    @property
    def device(self):
        """
        property decorator for device
        """
        return self.embeddings.src_embedding.weight.device
