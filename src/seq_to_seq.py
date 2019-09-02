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
       #TODO 
