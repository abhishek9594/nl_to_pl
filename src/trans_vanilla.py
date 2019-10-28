#!/usr/bin/env python
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from trans_encoder import TransEncoder
from trans_decoder import TransDecoder
from model_embeddings import ModelEmbeddings
from positional_embeddings import PositionalEmbeddings
from utils import subsequent_mask, clone

class TransVanilla(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate):
        super(TransVanilla, self).__init__()
        self.d_model = embed_size
        self.d_ff = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.embeddings = ModelEmbeddings(self.d_model, self.vocab)
        self.pe = PositionalEmbeddings(self.d_model, self.dropout_rate)
        
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
        P = F.log_softmax(tgt_decoded, dim=-1)

        tgt_padded_mask = (tgt_padded != self.vocab.tgt['<pad>']).float()
        #compute cross-entropy between tgt_words and tgt_predicted_words
        tgt_predicted = torch.gather(P, dim=-1, 
            index=tgt_padded[:, 1:].unsqueeze(-1)).squeeze(-1) * tgt_padded_mask[:, 1:]
        scores = tgt_predicted.sum(dim=0)
        return scores
        
    def encode(self, src, src_mask=None):
        x = self.pe(self.embeddings.src_embedding(src))
        for encoder in self.encoder_blocks:
            x = encoder(x, src_mask)
        return x

    def decode(self, src_encoded, tgt, src_mask=None, tgt_mask=None):
        x = self.pe(self.embeddings.tgt_embedding(tgt))
        for decoder in self.decoder_blocks:
            x = decoder(src_encoded, x, src_mask, tgt_mask)
        return self.vocab_project(x)

    @property
    def device(self):
        """
        property decorator for device
        """
        return self.embeddings.src_embedding.weight.device
    
    def save(self, path):
        """ 
        @param path (str): path to the model
        """
        params = {
            'args': dict(embed_size=self.embeddings.embed_size, 
            hidden_size=self.d_ff, vocab=self.vocab, dropout_rate=self.dropout_rate),
            'state_dict': self.state_dict()
        }
        torch.save(params, path)