#!/usr/bin/env python
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.trans_encoder import TransEncoder
from nn.trans_decoder import TransDecoder
from nn.model_embeddings import ModelEmbeddings
from nn.positional_embeddings import PositionalEmbeddings
from nn.utils import subsequent_mask

from trans_vanilla import TransVanilla

from lang.util import typename, parse_rule, type_of, extract_val_GenToken
from lang.py.grammar import is_builtin_type, is_terminal_ast_type, type_str_to_type

class TransCopy(TransVanilla):

    def __init__(self, embed_size, hidden_size, vocab, nodes, rules, dropout_rate):
        super(TransCopy, self).__init__(embed_size, hidden_size, vocab, nodes, rules, dropout_rate)

        self.context_proj = nn.Linear(self.d_model, 1)
        self.dec_in_proj = nn.Linear(self.d_model, 1)
        self.dec_out_proj = nn.Linear(self.d_model, 1)

    def forward(self, src_sents, tgt_tokens, tgt_rules, padx=0):
        """
        src_sents: (list[list[str]])
        tgt_tokens: (list[list[str]]) tgt_sents tokens containing nodes + GenTokens
        tgt_rules: (list[list[str]]) tgt_sents rules containing rules + GenTokens
        """
        src_tensor = self.vocab.src.sents2Tensor(src_sents).to(self.device)
        src_mask = (src_tensor != padx).unsqueeze(1)
        src_encoded = self.encode(src_tensor, src_mask)

        gen_toks_tensor, nodes_tensor = self.vocab.tgt_sents2Tensor(tgt_tokens).to(self.device), self.nodes.sents2Tensor(tgt_tokens).to(self.device)
        tgt_mask = ((gen_toks_tensor != padx) | (nodes_tensor != padx)).unsqueeze(1)
        subseq_mask = subsequent_mask(tgt_mask.shape[-1]).type_as(tgt_mask.data).to(self.device)
        tgt_mask = tgt_mask & subseq_mask
        tgt_encoded, q_key_src_dots = self.decode(src_encoded, gen_toks_tensor, nodes_tensor, src_mask, tgt_mask)

        gen_toks_decoded = self.gen_tok_project(tgt_encoded)
        P_gen_tok = F.softmax(gen_toks_decoded, dim=-1)

        rules_decoded = self.rule_project(tgt_encoded)
        log_P_rules = F.log_softmax(rules_decoded, dim=-1)

        src_ids_map_tgt = self.vocab.src_ids_in_tgt(src_sents, tgt_rules).to(self.device)
        max_unk_src_toks = self.vocab.max_unk_toks(src_sents)
        A_QK = torch.sum(torch.stack(q_key_src_dots, dim=0), dim=0) / 8 #normalize by num heads
        P_gen = self.p_gen(Y_e=src_encoded, A_QK=A_QK, gen_toks=gen_toks_tensor, nodes=nodes_tensor, Y_d=tgt_encoded)
        P_gen_copy = torch.cat((torch.mul(P_gen.unsqueeze(-1) , P_gen_tok), torch.zeros(P_gen_tok.shape[0], P_gen_tok.shape[1], max_unk_src_toks, device=self.device)), dim=-1)
        P_copy =  torch.mul((1 - P_gen).unsqueeze(-1), A_QK) #(b, Q, K)
        P_gen_copy = P_gen_copy.scatter_add(dim=-1, index=src_ids_map_tgt, source=P_copy)
                
        copy_mask = (P_gen_copy != 0)
        P_gen_copy = P_gen_copy.masked_fill(copy_mask == 0, 1.0)
        log_P_gen_copy = torch.log(P_gen_copy)
        #compute cross-entropy loss between copy_gen_toks and tgt_predicted_toks (log_p_gen_copy)
        copy_gen_toks = self.vocab.copy_gen_tok_ids(src_sents, tgt_rules).to(self.device)
        copy_gen_toks_mask = (copy_gen_toks != padx).float()
        loss_gen_toks = torch.gather(log_P_gen_copy, dim=-1,
            index=copy_gen_toks.unsqueeze(-1)).squeeze(-1) * copy_gen_toks_mask #(b, Q)

        #compute cross-entropy loss between tgt_rules and tgt_predicted_rules (log_P_rules)
        tgt_rules_tensor = self.rules.sents2Tensor(tgt_rules).to(self.device)
        tgt_rules_mask = (tgt_rules_tensor != padx).float()
        loss_rules = torch.gather(log_P_rules, dim=-1, 
            index=tgt_rules_tensor.unsqueeze(-1)).squeeze(-1) * tgt_rules_mask #(b, Q)

        loss = (loss_gen_toks + loss_rules).sum(dim=-1) #(b,)
        return loss

    def decode(self, src_encoded, gen_toks_tensor, nodes_tensor, src_mask=None, tgt_mask=None):
        x = self.pe(self.embeddings.gen_tok_embedding(gen_toks_tensor) + self.embeddings.node_embedding(nodes_tensor))
        for decoder in self.decoder_blocks:
            x, _, q_key_src_dots = decoder(src_encoded, x, src_mask, tgt_mask)
        return x, q_key_src_dots

    def p_gen(self, Y_e, A_QK, gen_toks, nodes, Y_d):
        X_d = self.pe(self.embeddings.gen_tok_embedding(gen_toks) + self.embeddings.node_embedding(nodes))
        return torch.sigmoid(self.context_proj(torch.bmm(A_QK, Y_e)) + self.dec_in_proj(X_d) + self.dec_out_proj(Y_d)).squeeze(-1) #(b, Q)

    def beam_search(self, src_sent, beam_size, max_decoding_time_step):
        """
        given a source sentence, search possible hyps, from this transformer model, up to the beam size
        @param src_sent (list[str]): source sentence
        @param beam_size (int)
        @param max_decoding_time_step (int): decode the hyp until <eos> or max decoding time step
        @return best_hyp (list[str]): best possible hyp
        """        
        src_tensor = self.vocab.src.sents2Tensor([src_sent]).to(self.device)
        src_encoded = self.encode(src_tensor, src_mask=None)

        t = 0

        unk_tok_ids = self.vocab.map_unk_src(src_sent)
        ids_unk_tok = {id : tok for tok, id in unk_tok_ids.items()} #inverse map idx => unk_tok
        max_unk_src_toks = len(ids_unk_tok)
        src_ids_map_tgt = self.vocab.src_ids_in_tgt([src_sent], [['dummy']]).to(self.device)[:, -1, :]

        explore_nodes = ['root']
        batch_toks = [] #input tokens
        hyp = []
        while len(explore_nodes):
            curr_node = explore_nodes.pop()
            batch_toks.append(curr_node)
            gen_toks_tensor = self.vocab.tgt_sents2Tensor([batch_toks]).to(self.device)
            nodes_tensor = self.nodes.sents2Tensor([batch_toks]).to(self.device)
            tgt_mask = subsequent_mask(nodes_tensor.shape[-1]).byte().to(self.device)
            tgt_encoded, q_key_src_dots = self.decode(src_encoded, gen_toks_tensor, nodes_tensor, src_mask=None, tgt_mask=tgt_mask)

            node_type = type_str_to_type(curr_node)

            if is_terminal_ast_type(node_type) or node_type == 'epsilon':
                hyp.append('GenToken[<eob>]')

            elif is_builtin_type(node_type):
                #GenToken
                gen_toks_decoded = self.gen_tok_project(tgt_encoded)
                P_gen_tok = F.softmax(gen_toks_decoded, dim=-1)[:, -1, :] #extract last gen tok
                A_QK = torch.sum(torch.stack(q_key_src_dots, dim=0), dim=0) / 8 #normalize by num heads
                P_gen = self.p_gen(Y_e=src_encoded, A_QK=A_QK, gen_toks=gen_toks_tensor, nodes=nodes_tensor, Y_d=tgt_encoded)[:, -1] #(b,)
                P_gen_copy = torch.cat((torch.mul(P_gen.unsqueeze(-1) , P_gen_tok), torch.zeros(P_gen_tok.shape[0], max_unk_src_toks, device=self.device)), dim=-1)
                P_copy =  torch.mul((1 - P_gen).unsqueeze(-1), A_QK[:, -1, :]) #(b, K)
                P_gen_copy = P_gen_copy.scatter_add(dim=-1, index=src_ids_map_tgt, source=P_copy)  

                gen_tok_mask = [0 if 'GenToken' not in self.vocab.tgt.id2word[i] else int(node_type == type_of(extract_val_GenToken(self.vocab.tgt.id2word[i]))) for i in range(len(self.vocab.tgt))]
                copy_mask = [int(node_type == type_of(ids_unk_tok[i])) for i in range(len(ids_unk_tok))]
                gen_tok_mask.extend(copy_mask)
                gen_copy_mask = ((P_gen_copy != 0) & (torch.tensor([gen_tok_mask]).byte().to(self.device)))

                P_gen_copy = P_gen_copy.masked_fill(gen_copy_mask == 0, 0.0)
                max_tok_id = torch.log(P_gen_copy).argmax().item()
                gen_tok = self.vocab.tgt.id2word[max_tok_id] if max_tok_id < len(self.vocab.tgt) else ids_unk_tok[max_tok_id - len(self.vocab.tgt)]
                hyp.extend([gen_tok, 'GenToken[<eob>]'])

            else:
                rules_decoded = self.rule_project(tgt_encoded) #(b, Q, R)
                out_rules = rules_decoded[:, -1, :] #(b, R)
                rules_mask = [[curr_node == rule_head for rule_head in self.rules.head_nodes()]]
                rules_mask = torch.tensor(rules_mask).byte().to(self.device)
                out_rules = out_rules.masked_fill(rules_mask == 0, -float('inf'))
                log_P_rules = F.log_softmax(out_rules, dim=-1)
                max_rule_id = log_P_rules.argmax().item()
                rule = self.rules.id2rule[max_rule_id]
                children_types_labels = parse_rule(rule)
                child_nodes = [typename(child_type) for (child_type, _) in children_types_labels]
                explore_nodes.extend(child_nodes[::-1]) #reverse due to DFS
                hyp.append(rule)
        return hyp
        """

        explore_nodes = [['root']]
        batch_toks = [['root']]
        hyps = []
        completed_hyps = []
        hyp_scores = torch.zeros(len(batch_toks), dtype=torch.float).to(self.device)

        while len(completed_hyps) < beam_size and t < max_decoding_time_step:
            num_hyp = len(explore_nodes)
            src_encoded_batch = src_encoded.expand(num_hyp, src_encoded.shape[1], src_encoded.shape[2])
            gen_toks_tensor = self.vocab.tgt_sents2Tensor(batch_toks).to(self.device)
            nodes_tensor = self.nodes.sents2Tensor(batch_toks).to(self.device)
            tgt_mask = subsequent_mask(nodes_tensor.shape[-1]).byte().to(self.device)
            tgt_encoded, q_key_src_dots = self.decode(src_encoded_batch, gen_toks_tensor, nodes_tensor, src_mask=None, tgt_mask=tgt_mask)

            log_P_rules = F.log_softmax(self.rule_project(tgt_encoded), dim=-1)

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
