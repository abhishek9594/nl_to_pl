#!/usr/bin/env python
"""
Seq2AST model
"""
import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F

from nn.model_embeddings import ModelEmbeddings

class Seq2AST(nn.Module):
    """
    Seq2AST model with attention
    BiLSTM encoder
    LSTM decoder
    """
    def __init__(self, embed_size, hidden_size, vocab, nodes, rules, dropout_rate):
        super(Seq2AST, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.nodes = nodes
        self.rules = rules
        self.embeddings = ModelEmbeddings(self.embed_size, self.vocab, self.nodes, self.rules)

        #initialize neural nets
        self.encoder = nn.LSTM(embed_size, self.hidden_size, bias=True, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTMCell(2*embed_size+self.hidden_size, self.hidden_size, bias=True)
        self.h_projection = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.c_projection = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.att_projection = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.combined_out_projection = nn.Linear(self.hidden_size*3, self.hidden_size, bias=False)
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.gen_tok_project = nn.Linear(self.hidden_size, len(self.vocab.tgt), bias=False)
        self.rule_project = nn.Linear(self.hidden_size, len(self.rules), bias=False)

    def forward(self, src_sents, tgt_nodes, tgt_tokens, tgt_actions, padx=0):
        """
        @param src_sents (list[list[str]]): batches of src sentences (list[str])
        @param tgt_nodes (list[list[str]]): batches of tgt nodes (list[str]) as the input to the decoder
        @param tgt_tokens (list[list[str]]): batches of tgt tokens (list[str]) as another input to the decoder
        @param tgt_actions (list[list[action]]): batches of actions as the gold output of the decoder
        @return scores (tensor(b,)): batch size tensor of the CE loss
        """
        src_in = self.vocab.src.sents2Tensor(src_sents).to(self.device) #(b, max_src_sent)
        src_lens = [len(src_sent) for src_sent in src_sents]
        src_mask = (src_in != padx).unsqueeze(1) #(b, 1, max_src_sent)
        src_encoded, dec_init_state = self.encode(src_in, src_lens)

        batch_size = len(tgt_nodes)
        tgt_in_nodes = self.nodes.nodes2Tensor(tgt_nodes).to(self.device)
        tgt_in_actions = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long, device=self.device),
                                    self.rules.rules2Tensor(tgt_actions).to(self.device)[:, :-1]], dim=-1) #(b, max_tgt_sent)
        tgt_in_tokens = self.vocab.tgt.sents2Tensor(tgt_tokens).to(self.device)
    
        tgt_encoded = self.decode(src_encoded, tgt_in_nodes, tgt_in_actions, tgt_in_tokens, dec_init_state, src_mask)

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

    def encode(self, src_in, src_lens):
        """
        apply the encoder on the source to obtain the encoder hidden states
        @param src_in (torch.tensor(b, max_src_len)): padded source sentences
        @param src_lens (list[int]): actual length of source sentences
        @return src_encoded (torch.tensor(b, max_src_len, 2*h)): sequence of encoder hidden states
        @return dec_init_state (tuple(Tensor, Tensor)): torch.tensor(b, h)
        """
        src_embed = self.embeddings.src_embedding(src_in)
        x = rnn.pack_padded_sequence(src_embed, src_lens, batch_first=True)
        src_enc_packed, (h_n, c_n) = self.encoder(x)
        src_encoded, src_lens_tensor = rnn.pad_packed_sequence(src_enc_packed, batch_first=True)
        #h_n.shape = (2, b, h)
        h_n_cat = torch.cat((h_n[0, :, :], h_n[1, :, :]), dim=-1).to(self.device)
        c_n_cat = torch.cat((c_n[0, :, :], c_n[1, :, :]), dim=-1).to(self.device)
        h_d = self.h_projection(h_n_cat)
        c_d = self.c_projection(c_n_cat)
        return src_encoded, (h_d, c_d)

    def decode(self, src_encoded, tgt_nodes, tgt_actions, tgt_tokens, dec_init_state, src_mask=None):   
        batch_size = src_encoded.shape[0]
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        y = torch.cat([self.embeddings.tgt_node_embedding(tgt_nodes), 
                        self.embeddings.tgt_action_embedding(tgt_actions) + self.embeddings.tgt_token_embedding(tgt_tokens)], dim=-1)

        (h_t, c_t) = dec_init_state

        src_encoded_proj = self.att_projection(src_encoded)

        out = []
        for y_t in torch.split(y, split_size_or_sections=1, dim=1):
            y_t = torch.squeeze(y_t, dim=1)#shape(b, 1, 2*e) -> (b, 2*e)
            i_t = torch.cat((y_t, o_prev), dim=-1) #(b, 2*e+h)
            (h_t, c_t), o_t = self.step(i_t, (h_t, c_t), src_encoded, src_encoded_proj, src_mask)
            out.append(o_t)
            o_prev = o_t

        decoder_out = torch.stack(out, dim=1) #(b, max_tgt_sent, h)
        return decoder_out

    def step(self, i_t, dec_state, src_encoded, src_encoded_proj, src_mask=None):
        """
        @param i_t (torch.tensor(b, 2*e+h)): decoder input at t
        @param dec_state (tuple(torch.tensor(b, h), torch.tensor(b, h)))
        @param src_encoded (torch.tensor(b, max_src_len, h*2))
        @param src_encoded_proj (torch.tensor(b, max_enc_len, h))
        @param src_mask (torch.tensor(b, max_enc_len))
        @return dec_next_state (tuple(torch.tensor(b, h), torch.tensor(b, h))): decoder next hidden and cell state
        @return o_t (torch.tensor(b, h)): decoder output at t
        """
        dec_next_state = self.decoder(i_t, dec_state)
        (h_t, c_t) = dec_next_state
        #attention scores
        e_t = torch.bmm(src_encoded_proj, h_t.unsqueeze(-1)).squeeze(-1) #(b, max_src_len)
        if src_mask is not None:
            e_t.masked_fill(src_mask == 0, -float('inf'))
        
        a_t = F.softmax(e_t, dim=-1) #(b, max_enc_len)
        o_t = torch.bmm(a_t.unsqueeze(1), src_encoded).squeeze(1) #(b, h*2)
        o_t = torch.cat((h_t, o_t), dim=-1) #(b, h*3)
        
        o_t = self.combined_out_projection(o_t) #(b, h)
        o_t = self.dropout_layer(torch.tanh(o_t))
        return dec_next_state, o_t

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

        src_in = self.vocab.src.sents2Tensor([src_sent]).to(self.device) #(b, max_src_sent)
        src_lens = [len(src_sent)]
        src_encoded, dec_init_state = self.encode(src_in, src_lens)

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
            tgt_encoded = self.decode(src_encoded, tgt_in_nodes, tgt_in_actions, tgt_in_tokens, dec_init_state, src_mask=None)
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
                tgt_actions.append(rule_pred)
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
        model = Seq2AST(**args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path):
        """ 
        @param path (str): path to the model
        """
        params = {
            'args': dict(embed_size=self.embeddings.embed_size, 
            hidden_size=self.hidden_size, vocab=self.vocab, nodes=self.nodes, rules=self.rules, dropout_rate=self.dropout_rate),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
