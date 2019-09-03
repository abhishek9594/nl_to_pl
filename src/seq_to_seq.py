#!/usr/bin/env python
"""
Seq2Seq model
"""
import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F

from model_embeddings import ModelEmbeddings

class Seq2Seq(nn.Module):
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
        self.decoder = nn.LSTMCell(embed_size+self.hidden_size, self.hidden_size, bias=True)
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
        finally run decoder to compute log-likelihood of target sentences
        @param source (list[list[str]]): list of source sentence tokens
        @param target (list[list[str]]): list of target sentence tokens, wrapped by <start> and <eos> tokens
        @return scores (torch.tensor(b, )): log-likelihood of generating gold target sentences
        """
        source_lengths = [len(s) for s in source]
    
        #convert list of sentences to tensors
        source_padded = self.vocab.src.sents2Tensor(source, device=self.device) #Tensor: (src_len, b)
        target_padded = self.vocab.tgt.sents2Tensor(target, device=self.device) #Tensor: (tgt_len, b)
        
        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)

        target_predicted = self.decode(target_padded, dec_init_state, enc_hiddens, enc_masks)
        P = F.log_softmax(target_predicted, dim=-1)

        #create mask to zero out probability for the pad tokens
        tgt_mask = (target_padded != self.vocab.tgt['<pad>']).float()
        #compute cross-entropy between tgt_words and tgt_predicted_words
        tgt_predicted_words_log_prob = torch.gather(P, dim=-1, 
            index=target_padded[1:].unsqueeze(-1)).squeeze(-1) * tgt_mask[1:]
        scores = tgt_predicted_words_log_prob.sum(dim=0)
        return scores

    def encode(self, source, source_lengths):
        """
        apply the encoder on the source to obtain the encoder hidden states
        @param source (torch.tensor(max_src_len, b)): padded source sentences
        @param source_lengths (list[int]): actual length of source sentences
        @return enc_hiddens (torch.tensor(b, max_src_len, 2*h)): sequence of encoder hidden states
        @return dec_init_state (tuple(Tensor, Tensor)): torch.tensor(b, h)
        """
        X = self.embeddings.src_embedding(source)
        X = rnn.pack_padded_sequence(X, source_lengths)
        enc_hiddens, (h_e, c_e) = self.encoder(X)
        enc_hiddens, src_lens_tensor = rnn.pad_packed_sequence(enc_hiddens)
        batch = source.shape[1]
        #h_e.shape = (2, b, h)
        h_e_cat = torch.cat((h_e[0, :, :], h_e[1, :, :]), dim=-1).to(self.device)
        c_e_cat = torch.cat((c_e[0, :, :], c_e[1, :, :]), dim=-1).to(self.device)
        #permute dim of enc_hiddens for batch_first
        enc_hiddens = enc_hiddens.permute(1, 0, 2)
        h_d = self.h_projection(h_e_cat)
        c_d = self.c_projection(c_e_cat)
        return enc_hiddens, (h_d, c_d)

    def decode(self, target, dec_init_state, enc_hiddens, enc_masks):
        """
        apply the decoder on target to predict target words
        @param target (torch.tensor(max_tgt_len , b)): padded target sentences
        @param dec_init_state (tuple(Tensor, Tensor)): torch.tensor(b, h)
        @param enc_hiddens (torch.tensor(b, max_src_len, 2*h))
        @param enc_masks (torch.tensor(b, max_src_len))
        @return tgt_predicted (torch.tensor(max_tgt_len-1, b, len(vocab.tgt)))
        """
        batch_size = enc_hiddens.shape[0]
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        #chop the <eos> token for max len tgt sentences
        target = target[:-1]
        
        Y = self.embeddings.tgt_embedding(target)

        (h_t, c_t) = dec_init_state

        enc_hiddens_proj = self.att_projection(enc_hiddens)

        hidden_outs = []
        for y_t in torch.split(Y, split_size_or_sections=1, dim=0):
            y_t = torch.squeeze(y_t, dim=0)#shape(1, b, e) -> (b, e)
            i_t = torch.cat((y_t, o_prev), dim=-1)
            (h_t, c_t), o_t = self.step(i_t, (h_t, c_t), enc_hiddens, enc_hiddens_proj, enc_masks)
            hidden_outs.append(o_t)
            o_prev = o_t

        hidden_outs = torch.stack(hidden_outs, dim=0)
        tgt_predicted = self.tgt_vocab_projection(hidden_outs)
        return tgt_predicted

    def step(self, i_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks):
        """
        @param i_t (torch.tensor(b, e+h)): decoder input at t
        @param dec_state (tuple(torch.tensor(b, h), torch.tensor(b, h)))
        @param enc_hiddens (torch.tensor(b, max_src_len, h*2))
        @param enc_hiddens_proj (torch.tensor(b, max_enc_len, h))
        @param enc_masks (torch.tensor(b, max_enc_len))
        @return dec_next_state (tuple(torch.tensor(b, h), torch.tensor(b, h))): decoder next hidden and cell state
        @return o_t (torch.tensor(b, h)): decoder output at t
        """
        dec_next_state = self.decoder(i_t, dec_state)
        (h_t, c_t) = dec_next_state
        #attention scores
        e_t = torch.bmm(enc_hiddens_proj, h_t.unsqueeze(-1)).squeeze(-1) #(b, max_src_len)
        #filling -inf to e_t where enc_masks has 1, to zero out <pad> toks
        #Note: e^{-inf} = 0
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))
        
        a_t = F.softmax(e_t, dim=-1) #(b, max_enc_len)
        o_t = torch.bmm(a_t.unsqueeze(1), enc_hiddens).squeeze(1) #(b, h*2)
        o_t = torch.cat((h_t, o_t), dim=-1) #(b, h*3)
        
        o_t = self.combined_out_projection(o_t) #(b, h)
        o_t = self.dropout(torch.tanh(o_t))
        return dec_next_state, o_t
    
    def generate_sent_masks(self, enc_hiddens, source_lengths):
        """
        generate sent masks for encoder hidden states
        @param enc_hiddens (torch.tensor(b, max_src_len, 2*h))
        @param source_lengths (list[int]): actual length of source sentences
        @return enc_masks (torch.tensor(b, max_src_len))
        """
        enc_masks = torch.zeros(enc_hiddens.shape[0], enc_hiddens.shape[1], dtype=torch.float, device=self.device)
        for i, src_len in enumerate(source_lengths):
            enc_masks[i, src_len:] = 1
        return enc_masks

    @property
    def device(self):
        """
        property decorator for device
        """
        return self.embeddings.src_embedding.weight.device

    @staticmethod
    def load(model_path):
        """ 
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = Seq2Seq(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path):
        """ 
        @param path (str): path to the model
        """
        params = {
            'args': dict(embed_size=self.embeddings.embed_size, 
            hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
