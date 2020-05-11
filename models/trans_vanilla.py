#!/usr/bin/env python
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.trans_encoder import TransEncoder
from nn.trans_decoder import TransDecoder
from nn.model_embeddings import ModelEmbeddings
from nn.positional_embeddings import PositionalEmbeddings
from nn.utils import subsequent_mask, clone

class TransVanilla(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, nodes, rules, dropout_rate):
        super(TransVanilla, self).__init__()
        self.d_model = embed_size
        self.d_ff = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.nodes = nodes
        self.rules = rules
        self.embeddings = ModelEmbeddings(self.d_model, self.vocab, self.nodes, self.rules)
        self.pe = PositionalEmbeddings(self.d_model, self.dropout_rate)
        self.scale_decoder_embedding = nn.Linear(self.d_model*2, self.d_model, bias=False)
        
        self.encoder_blocks = clone(TransEncoder(self.d_model, self.d_ff, self.dropout_rate), n=6)
        self.decoder_blocks = clone(TransDecoder(self.d_model, self.d_ff, self.dropout_rate), n=6)
        self.gen_tok_project = nn.Linear(self.d_model, len(self.vocab.tgt), bias=False)
        self.rule_project = nn.Linear(self.d_model, len(self.rules), bias=False)

    def forward(self, src_sents, tgt_nodes, tgt_tokens, tgt_actions, padx=0):
        """
        @param src_sents (list[list[str]]): batches of src sentences (list[str])
        @param tgt_nodes (list[list[str]]): batches of tgt nodes (list[str]) as the input to the decoder
        @param tgt_tokens (list[list[str]]): batches of tgt tokens (list[str]) as another input to the decoder
        @param tgt_actions (list[list[action]]): batches of actions as the gold output of the decoder
        @return scores (tensor(b,)): batch size tensor of the CE loss
        """
        src_in = self.vocab.src.sents2Tensor(src_sents).to(self.device) #(b, max_src_sent)
        src_mask = (src_in != padx).unsqueeze(1) #(b, 1, max_src_sent)
        src_encoded = self.encode(src_in, src_mask)

        batch_size = len(tgt_nodes)
        tgt_in_nodes = self.nodes.nodes2Tensor(tgt_nodes).to(self.device)
        tgt_in_actions = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long, device=self.device),
                                    self.rules.rules2Tensor(tgt_actions).to(self.device)[:, :-1]], dim=-1) #(b, max_tgt_sent)
        tgt_in_tokens = self.vocab.tgt.sents2Tensor(tgt_tokens).to(self.device)
        tgt_mask = (tgt_in_nodes != padx).unsqueeze(1)
        subseq_mask = subsequent_mask(tgt_in_nodes.shape[-1]).type_as(tgt_mask.data).to(self.device)
        tgt_mask = tgt_mask & subseq_mask
        tgt_encoded = self.decode(src_encoded, tgt_in_nodes, tgt_in_actions, tgt_in_tokens, src_mask, tgt_mask)

        tgt_rules_pred = F.log_softmax(self.rule_project(tgt_encoded), dim=-1)
        tgt_rules_idx = self.rules.rules2Tensor(tgt_actions).to(self.device) #(b, max_tgt_sent)
        tgt_rules_mask = (tgt_rules_idx != padx).float()
        loss_rules = torch.gather(tgt_rules_pred, dim=-1,
                                    index=tgt_rules_idx.unsqueeze(-1)).squeeze(-1) * tgt_rules_mask
        
        tgt_toks_pred = F.log_softmax(self.gen_tok_project(tgt_encoded), dim=-1)
        tgt_toks_idx = self.vocab.tgt.sents2Tensor(tgt_actions).to(self.device) #(b, max_tgt_sent)
        tgt_toks_mask = (tgt_toks_idx != padx).float()
        loss_gen_toks = torch.gather(tgt_toks_pred, dim=-1,
                                    index=tgt_toks_idx.unsqueeze(-1)).squeeze(-1) * tgt_toks_mask

        scores = (loss_rules + loss_gen_toks).sum(dim=-1) #(b,)
        return scores
        
    def encode(self, src, src_mask=None):
        x = self.pe(self.embeddings.src_embedding(src))
        for encoder in self.encoder_blocks:
            x, _ = encoder(x, src_mask)
        return x

    def decode(self, src_encoded, tgt_nodes, tgt_actions, tgt_tokens, src_mask=None, tgt_mask=None):     
        y2 = torch.cat([self.pe(self.embeddings.tgt_node_embedding(tgt_nodes)),
                    self.pe( self.embeddings.tgt_action_embedding(tgt_actions) + self.embeddings.tgt_token_embedding(tgt_tokens))], dim=-1)
        y = self.scale_decoder_embedding(y2)
        for decoder in self.decoder_blocks:
            y, _, _ = decoder(src_encoded, y, src_mask, tgt_mask)
        return y

    def generate(self, lang, src_sent):
        """
        given a source sentence, search possible hyps, from this transformer model
        @param lang (str): target language
        @param src_sent (list[str]): source sentence
        @return actions (list[action]): decoded AST actions
        """
        
        if lang == 'lambda':
            from lang.Lambda.asdl import ASDLGrammar
            from lang.Lambda.transition_system import ApplyRuleAction, GenTokenAction, ReduceAction

            asdl_desc = open('lang/Lambda/lambda_asdl.txt').read()
            grammar = ASDLGrammar.from_text(asdl_desc)            
        else:
            print('language: %s currently not supported' % (lang))
            return

        src_in = self.vocab.src.sents2Tensor([src_sent]).to(self.device)
        src_encoded = self.encode(src_in, src_mask=None)

        explore_nodes = ['<start>']
        tgt_nodes, tgt_actions, tgt_tokens = [], ['<pad>'], ['<pad>']
        actions = []
        while len(explore_nodes) > 0:
            if grammar.mul_cardinality(explore_nodes[-1]):
                curr_node = explore_nodes[-1]
            else:
                curr_node = explore_nodes.pop()
            tgt_nodes.append(curr_node)

            tgt_in_nodes = self.nodes.nodes2Tensor([tgt_nodes]).to(self.device)
            tgt_in_actions = self.rules.rules2Tensor([tgt_actions]).to(self.device)
            tgt_in_tokens = self.vocab.tgt.sents2Tensor([tgt_tokens]).to(self.device)
            tgt_mask = subsequent_mask(tgt_in_nodes.shape[-1]).byte().to(self.device)
            tgt_encoded = self.decode(src_encoded, tgt_in_nodes, tgt_in_actions, tgt_in_tokens, src_mask=None, tgt_mask=tgt_mask)
            if grammar.node_prim_type(curr_node):
                tgt_toks_pred = F.log_softmax(self.gen_tok_project(tgt_encoded), dim=-1)[:, -1, :] #extract last pred token
                top_tok_id = tgt_toks_pred.argmax().item()
                actions.append(GenTokenAction(self.vocab.tgt.id2word[top_tok_id]))
                tgt_actions.append('<pad>')
                tgt_tokens.append(self.vocab.tgt.id2word[top_tok_id])
            else:
                #composite_type => rule
                rules_out = self.rule_project(tgt_encoded)[:, -1, :] #(b, R)
                rules_mask = torch.tensor([self.rules.rule_match(curr_node)]).byte().to(self.device)
                rules_cand = rules_out.masked_fill(rules_mask == 0, -float('inf'))
                tgt_rules_pred = F.log_softmax(rules_cand, dim=-1)
                top_rule_id = tgt_rules_pred.argmax().item()
                rule_pred = self.rules.id2rule[top_rule_id]
                if rule_pred == 'Reduce':
                    actions.append(ReduceAction())
                    explore_nodes.pop()
                else:
                    actions.append(ApplyRuleAction(rule_pred))
                    #extract next action nodes from rule_pred constructor
                    fields = rule_pred.constructor.fields
                    action_nodes = []
                    for field in fields: #Field(name, ASDLType(name), cardinality)
                        node_name = field.type.name
                        field_cardinality = field.cardinality
                        if field_cardinality == 'multiple':
                            node_name += '*'
                        action_nodes.append(node_name)
                    explore_nodes.extend(action_nodes[::-1])
                tgt_actions.append(rule_pred)
                tgt_tokens.append('<pad>')
        
        return actions

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
            hidden_size=self.d_ff, vocab=self.vocab, nodes=self.nodes, rules=self.rules, dropout_rate=self.dropout_rate),
            'state_dict': self.state_dict()
        }
        torch.save(params, path)
