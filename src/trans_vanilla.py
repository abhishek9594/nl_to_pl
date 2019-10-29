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

        tgt_padded_mask = (tgt_padded != padx).float()
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

    def beam_search(self, src_sent, beam_size, max_decoding_time_step):
        """
        given a source sentence, search possible hyps, from this transformer model, up to the beam size
        @param src_sent (list[str]): source sentence
        @param beam_size (int)
        @param max_decoding_time_step (int): decode the hyp until <eos> or max decoding time step
        @return best_hyp (list[str]): best possible hyp
        """
        src_padded = self.vocab.src.sents2Tensor([src_sent]).to(self.device)
        src_encoded = self.encode(src_padded, src_mask=None)

        t = 0

        hyps = [['<start>']]
        completed_hyps = []
        hyp_scores = torch.zeros(len(hyps), dtype=torch.float).to(self.device)

        while len(completed_hyps) < beam_size and t < max_decoding_time_step:
            num_hyp = len(hyps)
            src_encoded_batch = src_encoded.expand(num_hyp, src_encoded.shape[1], src_encoded.shape[2])
            x = self.vocab.tgt.sents2Tensor(hyps).to(self.device)
            tgt_mask = subsequent_mask(x.shape[-1]).byte().to(self.device)
            tgt_decoded = self.decode(src_encoded_batch, x, src_mask=None, tgt_mask=tgt_mask)
            P = F.log_softmax(tgt_decoded, dim=-1)[:, -1, :] #extract last predicted word
            
            num_live_hyp = beam_size - len(completed_hyps)
            live_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(P) + P).view(-1) #shape = (num_live_hyp * len(vocab.tgt))
            top_word_scores, top_word_pos = torch.topk(live_hyp_scores, k=num_live_hyp)

            prev_hyp_ids = top_word_pos / len(self.vocab.tgt)
            hyp_word_ids = top_word_pos % len(self.vocab.tgt)

            new_hyps = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, top_word_score in zip(prev_hyp_ids, hyp_word_ids, top_word_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                top_word_score = top_word_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hyps[prev_hyp_id] + [hyp_word]
                if hyp_word == '<eos>':
                    completed_hyps.append((new_hyp_sent[1:-1], top_word_score))
                else:
                    new_hyps.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(top_word_score)

            hyps = new_hyps
            
            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long).to(self.device)
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float).to(self.device)

            t += 1
        #end-while
        #in this case best_hyp is not guaranteed
        if len(completed_hyps) == 0:
            completed_hyps.append((hyps[0][1:], hyp_scores[0].item()))

        completed_hyps.sort(key=lambda (hyp, score): score, reverse=True)
        best_hyp = [str(word) for word in completed_hyps[0][0]]
        return best_hyp

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
        model = TransVanilla(**args)
        model.load_state_dict(params['state_dict'])
        return model
    
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