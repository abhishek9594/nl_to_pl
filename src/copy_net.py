#!/usr/bin/env python
"""
CopyNet model
"""
import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F

from model_embeddings import ModelEmbeddings

class CopyNet(nn.Module):
    """
    seq2seq model with attention &
    copy net mechanism
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
        super(CopyNet, self).__init__()
        self.embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        #initialize neural nets
        self.encoder = nn.LSTM(embed_size, self.hidden_size, bias=True, bidirectional=True)
        self.decoder = nn.LSTMCell(embed_size+self.hidden_size*2, self.hidden_size, bias=True)
        self.h_projection = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.c_projection = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.att_projection = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.copy_projection = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
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
        target_copy_padded = self.vocab.tgt.tgt_sents2Tensor(source, target, device=self.device) #Tensor: (tgt_len, b)
        src_tgt_ids, max_unk_src_words = self.vocab.map_src_tgt(source, device=self.device) #Tensor: (b, src_len)
        
        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)

        target_predicted = self.decode(target_padded, src_tgt_ids, max_unk_src_words, dec_init_state, enc_hiddens, enc_masks)
        P = torch.log(target_predicted)

        #create mask to zero out probability for the pad tokens
        tgt_mask = (target_copy_padded != self.vocab.tgt['<pad>']).float()
        #compute cross-entropy between tgt_words and tgt_predicted_words
        tgt_predicted_words_log_prob = torch.gather(P, dim=-1, 
            index=target_copy_padded[1:].unsqueeze(-1)).squeeze(-1) * tgt_mask[1:]
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

    def decode(self, target, src_tgt_ids, max_unk_src_words, dec_init_state, enc_hiddens, enc_masks):
        """
        apply the decoder on target to predict target words
        @param target (torch.tensor(max_tgt_len , b)): padded target sentences
        @param src_tgt_ids (torch.tensor(b, max_src_len)): indices of src words mapping to target vocab
        @param max_unk_src_words (int): maximum number of unk src words in target vocab
        @param dec_init_state (tuple(Tensor, Tensor)): torch.tensor(b, h)
        @param enc_hiddens (torch.tensor(b, max_src_len, 2*h))
        @param enc_masks (torch.tensor(b, max_src_len))
        @return tgt_predicted (torch.tensor(max_tgt_len-1, b, len(vocab.tgt)))
        """
        #chop the <eos> token for max len tgt sentences
        target = target[:-1]
        
        Y = self.embeddings.tgt_embedding(target)

        (h_t, c_t) = dec_init_state

        enc_hiddens_proj = self.att_projection(enc_hiddens)

        outs = []
        for y_t in torch.split(Y, split_size_or_sections=1, dim=0):
            y_t = torch.squeeze(y_t, dim=0)#shape(1, b, e) -> (b, e)
            o_t, (h_t, c_t) = self.step(y_t, (h_t, c_t), src_tgt_ids, max_unk_src_words, enc_hiddens, enc_hiddens_proj, enc_masks)
            outs.append(o_t)

        tgt_predicted = torch.stack(outs, dim=0)
        return tgt_predicted

    def step(self, y_t, dec_state, src_tgt_ids, max_unk_src_words, enc_hiddens, enc_hiddens_proj, enc_masks):
        """
        @param y_t (torch.tensor(b, e)): decoder embedding input at time step t
        @param dec_state (tuple(torch.tensor(b, h), torch.tensor(b, h)))
        @param src_tgt_ids (torch.tensor(b, max_src_len)): indices of src words mapping to target vocab
        @param max_unk_src_words (int): maximum number of unk src words in target vocab
        @param enc_hiddens (torch.tensor(b, max_src_len, h*2))
        @param enc_hiddens_proj (torch.tensor(b, max_enc_len, h))
        @param enc_masks (torch.tensor(b, max_enc_len))
        @return o_t (torch.tensor(b, h)): decoder output at t
        @return dec_next_state (tuple(torch.tensor(b, h), torch.tensor(b, h))): decoder next hidden and cell state
        """
        (h_t, c_t) = dec_state
        #attention scores
        e_t = torch.bmm(enc_hiddens_proj, h_t.unsqueeze(-1)).squeeze(-1) #(b, max_src_len)
        #filling -inf to e_t where enc_masks has 1, to zero out <pad> toks
        #Note: e^{-inf} = 0
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))
        
        a_t = F.softmax(e_t, dim=-1) #(b, max_enc_len)
        att_read_t = torch.bmm(a_t.unsqueeze(1), enc_hiddens).squeeze(1) #(b, h*2)
        i_t = torch.cat((att_read_t, y_t), dim=-1) #(b, e+h*2)
        dec_next_state = self.decoder(i_t, dec_state) #((h_t+1, c_t+1): ((b, h), (b, h)))
        (h_t_next, c_t_next) = dec_next_state

        gen_t = self.tgt_vocab_projection(h_t_next) #(b, |T_V|)
        gen_t = torch.exp(gen_t)
        batch_size = gen_t.shape[0]
        copy_t = torch.zeros(batch_size, max_unk_src_words, device=self.device)
        o_t = torch.cat((gen_t, copy_t), dim=-1) #(b, |T_V|+max_unk_src_words)
        max_src_len = enc_hiddens.shape[1]
        for i in range(max_src_len):
            copy_t_i = torch.exp(torch.bmm(torch.tanh(self.copy_projection(enc_hiddens[:, i])).unsqueeze(1), h_t_next.unsqueeze(-1)).view(-1)) #(b, )
            if enc_masks is not None:
                copy_t_i.data.masked_fill_(enc_masks[:, i].byte(), 0)
            o_t.scatter_add_(dim=-1, index=src_tgt_ids[:, i].unsqueeze(1), src=copy_t_i.unsqueeze(1))
        o_t.div_(torch.sum(o_t, dim=-1).unsqueeze(1))

        return o_t, dec_next_state
    
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

    def beam_search(self, src_sent, beam_size, max_decoding_time_step):
        """
        given a source sentence, search possible hyps, from this seq-2seq model, up to the beam size
        @param src_sent (list[str]): source sentence
        @param beam_size (int)
        @param max_decoding_time_step (int): decode the hyp until <eos> or max decoding time step
        @return best_hyp (list[str]): best possible hyp
        """
        source = [src_sent]
        source_lengths = [len(src_sent)]
        source_padded = self.vocab.src.sents2Tensor(source, device=self.device)
        src_tgt_ids, max_unk_src_words = self.vocab.map_src_tgt(source, device=self.device)

        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_hiddens_proj = self.att_projection(enc_hiddens)

        t = 0

        hyps = [['<start>']]
        completed_hyps = []
        hyp_scores = torch.zeros(len(hyps), dtype=torch.float, device=self.device)

        (h_t, c_t) = dec_init_state
        while len(completed_hyps) < beam_size and t < max_decoding_time_step:
            num_hyp = len(hyps)
            y_t = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hyps], dtype=torch.long, device=self.device)
            y_t = self.embeddings.tgt_embedding(y_t)
            enc_hiddens_batch = enc_hiddens.expand(num_hyp, enc_hiddens.shape[1], enc_hiddens.shape[2])
            enc_hiddens_proj_batch = enc_hiddens_proj.expand(num_hyp, enc_hiddens_proj.shape[1], enc_hiddens_proj.shape[2])
            o_t, (h_t, c_t) = self.step(y_t, (h_t, c_t), src_tgt_ids, max_unk_src_words, enc_hiddens_batch, enc_hiddens_proj_batch, enc_masks=None)

            log_p_t = o_t
            
            num_live_hyp = beam_size - len(completed_hyps)
            live_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1) #shape = (num_live_hyp * len(vocab.tgt))
            top_word_scores, top_word_pos = torch.topk(live_hyp_scores, k=num_live_hyp)

            prev_hyp_ids = top_word_pos / (len(self.vocab.tgt) + max_unk_src_words)
            hyp_word_ids = top_word_pos % (len(self.vocab.tgt) + max_unk_src_words)

            new_hyps = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, top_word_score in zip(prev_hyp_ids, hyp_word_ids, top_word_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                top_word_score = top_word_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id] if hyp_word_id < len(self.vocab.tgt) else source[0][hyp_word_id - len(self.vocab.tgt)]
                new_hyp_sent = hyps[prev_hyp_id] + [hyp_word]
                if hyp_word == '<eos>':
                    completed_hyps.append((new_hyp_sent[1:-1], top_word_score))
                else:
                    new_hyps.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(top_word_score)

            hyps = new_hyps
            
            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            
            h_t, c_t = h_t[live_hyp_ids], c_t[live_hyp_ids]

            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

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
        model = CopyNet(vocab=params['vocab'], **args)
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
