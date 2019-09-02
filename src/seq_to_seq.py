#!/usr/bin/env python
"""
Seq2Seq model
"""
import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F

from model_embeddings import ModelEmbeddings

class Seq2Seq(nn.Module)
    """
    Seq2Seq model with attention
    BiLSTM encoder
    LSTM decoder
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate):
        """
        @param embed_size (int): embedding size
        @param hidden_size (int): hidden size
        @param vocab (Vocab): obj containing src and tgt VocabEntry
        @param dropout_rate (float): dropout probability
        """
        super(Seq2Seq, self).__init__()
        self.embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        #initialize neural nets
        self.encoder = nn.LSTM(embed_size, self.hidden_size, bias=True, bidirectional=True)
        self.decoder = nn.LSTMCell(embed_size, self.hidden_size, bias=True)
        self.h_projection = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.c_projection = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.att_projection = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.combined_out_projection = nn.Linear(self.hidden_size*3, self.hidden_size, bias=False)
        self.tgt_vocab_projection = nn.Linear(self.hidden_size, len(self.vocab.tgt), bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, source, target):
        """
        given a batch of source and target sentences, run encoder on source
        to compute the init hidden state for decoder
        run decoder to compute log-likelihood of target sentences 
        @param source (list[list[str]]): list of source sentence tokens
        @param target (list[list[str]]): list of target sentence tokens, 
            wrapped by <start> and <eos> tokens
        @return scores (Tensor): Tensor of shape (b, ) representing the
            log-likelihood of generating gold target sentences
            where b = batch size.
        """
        source_lengths = [len(s) for s in source]
    
        #convert list of sentences to tensors
        source_padded = self.vocab.src.sents2Tensor(source, device=self.device) #Tensor: (src_len, b)
        target_padded = self.vocab.src.sents2Tensor(target, device=self.device) #Tensor: (tgt_len, b)
        #TODO

    @property
    def device(self):
        """
        property decorator for device
        """
        return self.embeddings.src_embedding.weight.device
