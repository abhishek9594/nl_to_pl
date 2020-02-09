#!/usr/bin/env python
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from trans_encoder import TransEncoder
from trans_decoder import TransDecoder
from model_embeddings import ModelEmbeddings
from positional_embeddings import PositionalEmbeddings
from utils import src_to_tensor, tgt_to_tensors, map_src_tgt_tokens, tgt_to_rules
from utils import subsequent_mask

from trans_vanilla import TransVanilla

class TransCopy(TransVanilla):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate):
        super(TransCopy, self).__init__(embed_size, hidden_size, vocab, dropout_rate)

        self.context_proj = nn.Linear(self.d_model, 1)
        self.dec_in_proj = nn.Linear(self.d_model, 1)
        self.dec_out_proj = nn.Linear(self.d_model, 1)

    def forward(self, src, tgt, padx=0):
        """
        src: (list[list[str]])
        tgt: (list[list[str]])
        """
        src_tensor = src_to_tensor(src, self.vocab.src, device=self.device)
        src_mask = (src_tensor != padx).unsqueeze(1)
        src_encoded = self.encode(src_tensor, src_mask)

        gen_toks_tensor, nodes_tensor = tgt_to_tensors(tgt, self.vocab.tgt, device=self.device)
        tgt_mask = ((gen_toks_tensor != padx) | (nodes_tensor != padx)).unsqueeze(1)
        subseq_mask = subsequent_mask(tgt_mask.shape[-1]).type_as(tgt_mask.data).to(self.device)
        tgt_mask = tgt_mask & subseq_mask
        tgt_encoded, q_key_src_dots = self.decode(src_encoded, gen_toks_tensor, nodes_tensor, src_mask, tgt_mask)

        gen_toks_decoded = self.gen_tok_project(tgt_encoded)
        P_gen_tok = F.softmax(gen_toks_decoded, dim=-1)

        rules_decoded = self.rule_project(tgt_encoded)
        log_P_rules = F.log_softmax(rules_decoded, dim=-1)

        tgt_gen_toks, src_ids_map_tgt, max_unk_src_words = map_src_tgt_tokens(src, tgt, self.vocab, self.device)
        A_QK = torch.sum(torch.stack(q_key_src_dots, dim=0), dim=0) / 8 #normalize by num heads
        P_gen = self.p_gen(Y_e=src_encoded, A_QK=A_QK, gen_toks=gen_toks_tensor, nodes=nodes_tensor, Y_d=tgt_encoded)
        P_gen_copy = torch.cat((torch.mul(P_gen.unsqueeze(-1) , P_gen_tok), torch.zeros(P_gen_tok.shape[0], P_gen_tok.shape[1], max_unk_src_words, device=self.device)), dim=-1)
        P_copy =  torch.mul((1 - P_gen).unsqueeze(-1), A_QK) #(b, Q, K)
        P_gen_copy = P_gen_copy.scatter_add(dim=-1, index=src_ids_map_tgt, source=P_copy)
                
        copy_mask = (P_gen_copy != 0)
        P_gen_copy = P_gen_copy.masked_fill(copy_mask == 0, 1.0)
        log_P_gen_copy = torch.log(P_gen_copy)
        #compute cross-entropy loss between tgt_gen_toks and tgt_predicted_toks (log_p_gen_copy)
        tgt_gen_toks_mask = (tgt_gen_toks != padx).float()
        loss_gen_toks = torch.gather(log_P_gen_copy, dim=-1,
            index=tgt_gen_toks.unsqueeze(-1)).squeeze(-1) * tgt_gen_toks_mask #(b, Q)

        #compute cross-entropy loss between tgt_rules and tgt_predicted_rules (log_P_rules)
        tgt_rules = tgt_to_rules(tgt, padx).to(self.device)
        tgt_rules_mask = (tgt_rules != padx).float()
        loss_rules = torch.gather(log_P_rules, dim=-1, 
            index=tgt_rules.unsqueeze(-1)).squeeze(-1) * tgt_rules_mask #(b, Q)

        loss = (loss_gen_toks + loss_rules).sum(dim=-1) #(b,)
        return loss

    def decode(self, src_encoded, gen_toks_tensor, nodes_tensor, src_mask=None, tgt_mask=None):
        x = self.pe(self.embeddings.gen_tok_embedding(gen_toks_tensor) + self.embeddings.node_embedding(nodes_tensor))
        for decoder in self.decoder_blocks:
            x, _, q_key_src_dots = decoder(src_encoded, x, src_mask, tgt_mask)
        return x, q_key_src_dots

    def p_gen(self, Y_e, A_QK, gen_toks, nodes, Y_d):
        X_d = self.pe(self.embeddings.gen_tok_embedding(gen_toks) + self.embeddings.node_embedding(nodes))
        return torch.sigmoid(self.context_proj(torch.bmm(A_QK, Y_e)) + self.dec_in_proj(X_d) + self.dec_out_proj(Y_d)).squeeze(-1)

    def beam_search(self, src_sent, beam_size, max_decoding_time_step):
        """
        given a source sentence, search possible hyps, from this transformer model, up to the beam size
        @param src_sent (list[str]): source sentence
        @param beam_size (int)
        @param max_decoding_time_step (int): decode the hyp until <eos> or max decoding time step
        @return best_hyp (list[str]): best possible hyp
        """
        """
        src_padded = self.vocab.src.sents2Tensor([src_sent]).to(self.device)
        src_encoded = self.encode(src_padded, src_mask=None)

        t = 0

        unk_word_idx = map_src_words_tgt(src_sent, self.vocab)
        idx_unk_word = {idx : word for word, idx in unk_word_idx.items()} #inverse map idx => unk_word
        max_unk_src_words = len(idx_unk_word)
        src_ids_map_tgt = [self.vocab.tgt[word] if word not in unk_word_idx else len(self.vocab.tgt) + unk_word_idx[word] for word in src_sent]
        src_ids_map_tgt = torch.tensor(src_ids_map_tgt, dtype=torch.long, device=self.device)

        hyps = [['<start>']]
        completed_hyps = []
        hyp_scores = torch.zeros(len(hyps), dtype=torch.float).to(self.device)

        while len(completed_hyps) < beam_size and t < max_decoding_time_step:
            num_hyp = len(hyps)
            src_encoded_batch = src_encoded.expand(num_hyp, src_encoded.shape[1], src_encoded.shape[2])
            x = self.vocab.tgt.sents2Tensor(hyps).to(self.device)
            tgt_mask = subsequent_mask(x.shape[-1]).byte().to(self.device)
            tgt_encoded, q_key_src_dots = self.decode(src_encoded_batch, x, src_mask=None, tgt_mask=tgt_mask)

            tgt_decoded = self.vocab_project(tgt_encoded)
            P_vocab = F.softmax(tgt_decoded, dim=-1)[:, -1, :] #extract last generated word

            src_ids_map_tgt_batch = src_ids_map_tgt.expand(num_hyp, src_ids_map_tgt.shape[0])
            A_QK = torch.sum(torch.stack(q_key_src_dots, dim=0), dim=0) / 8 #normalize by num heads
            P_gen = self.p_gen(Y_e=src_encoded_batch, A_QK=A_QK, tgt=x, Y_d=tgt_encoded)[:, -1] #(b,)
            P_gen_copy = torch.cat((torch.mul(P_gen.unsqueeze(-1) , P_vocab), torch.zeros(P_vocab.shape[0], max_unk_src_words, device=self.device)), dim=-1)
            P_copy =  torch.mul((1 - P_gen).unsqueeze(-1), A_QK[:, -1, :]) #(b, K)
            P_gen_copy = P_gen_copy.scatter_add(dim=-1, index=src_ids_map_tgt_batch, source=P_copy)
            
            copy_mask = (P_gen_copy != 0)
            P_gen_copy = P_gen_copy.masked_fill(copy_mask == 0, 1.0)
            P = torch.log(P_gen_copy)
            
            num_live_hyp = beam_size - len(completed_hyps)
            live_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(P) + P).view(-1) #shape = (num_live_hyp * len(vocab.tgt) + max_unk_src_words)
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

                hyp_word = self.vocab.tgt.id2word[hyp_word_id] if hyp_word_id < len(self.vocab.tgt) else idx_unk_word[hyp_word_id - len(self.vocab.tgt)]
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
        """
        pass
    
    @staticmethod
    def load(model_path):
        """ 
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = TransCopy(**args)
        model.load_state_dict(params['state_dict'])
        return model