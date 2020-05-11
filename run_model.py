#!/usr/bin/env python
"""
run script to train and test our neural model

Usage:
    run_model.py train --model-type=<str> --lang=<str> --train-data=<file> --dev-data=<file> --vocab=<file> --nodes=<file> --rules=<file> [options]
    run_model.py transfer MODEL_PATH --model-type=<str> --lang=<str> --train-data=<file> --dev-data=<file> --vocab=<file> --nodes=<file> --rules=<file> [options]
    run_model.py test --model-type=<str> --lang=<str> MODEL_PATH TEST_FILE OUTPUT_FILE [options]

Options:
    -h --help                       show this screen.
    --model-type=<str>              neural model to run
    --lang=<str>                    target language
    --train-data=<file>             train data file
    --dev-data=<file>               dev data file
    --vocab=<file>                  vocab file
    --nodes=<file>                  node file
    --rules=<file>                  rule file
    --batch-size=<int>              batch size [default: 8]
    --embed-size=<int>              embedding size [default: 512]
    --hidden-size=<int>             hidden size [default: 2048]
    --max-epoch=<int>               max epoch [default: 10]
    --patience=<int>                num epochs early stopping [default: 2]
    --dropout=<float>               dropout rate [default: 0.1]
    --lr=<float>                    learning rate [default: 1e-4]
    --save-model-name=<file>        save model name
"""
from __future__ import division

import time, math
from docopt import docopt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from vocab import Vocab
from node import Node
from rule import Rule
from utils import read_corpus, batch_iter, save_sents, comp_exact_match

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(model, dev_src, dev_tgt, lang, batch_size=32):
    """
    validate model on dev set
    @param model
    @param dev_src (list(list[str])): list of source sentences (list of tokens)
    @param dev_tgt (list[str]): list of target sentences
    @param lang: target language
    @return dev_loss (float): cross entropy loss on dev set
    """
    was_training = model.training
    model.eval()
    
    cum_loss = .0
    cum_tgt_words = 0

    with torch.no_grad():
        for src_sents, tgt_nodes, tgt_tokens, tgt_actions in batch_iter(dev_src, dev_tgt, lang, batch_size):
            num_words_to_predict = sum(len(actions) for actions in tgt_actions)
            loss = -model(src_sents, tgt_nodes, tgt_tokens, tgt_actions).sum()
            
            cum_loss += loss
            cum_tgt_words += num_words_to_predict
        
        dev_loss = cum_loss / cum_tgt_words
    
    if was_training:
        model.train()

    return dev_loss

def decode(model, lang, test_src):
    """
    run inference on model to generate target sentences
    @param model (TransVanilla)
    @param lang (str): target language
    @param test_src (list[list[str]): list of test source sentences
    @return tgt_actions (list[list[action]]): list of AST actions
    """
    was_training = model.training
    model.eval()

    tgt_actions = []
    with torch.no_grad():
        for src_sent in test_src:
            actions = model.generate(lang, src_sent)
            tgt_actions.append(actions)

    if was_training:
        model.train
    return tgt_actions

def train(args,model):
    """
    train our neural model
    @param args (dict): command line args
    @param model: neural model to train
    """
    lang = args['--lang']

    train_src, train_tgt = read_corpus(args['--train-data'])

    dev_src, dev_tgt = read_corpus(args['--dev-data'])

    train_batch_size = dev_batch_size = int(args['--batch-size'])
    model_save_path = 'models/' + lang + '_' + args['--save-model-name']
    model.train()
    model = model.to(device)

    init_lr = float(args['--lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.98), eps=1e-09)

    cum_loss = .0
    cum_tgt_words = 0
    hist_dev_losses = []
    patience = 0

    begin_time = time.time()
    for epoch in range(int(args['--max-epoch'])):
        for src_sents, tgt_nodes, tgt_tokens, tgt_actions in batch_iter(train_src, train_tgt, lang, batch_size=train_batch_size, shuffle=True):

            num_words_to_predict = sum(len(actions) for actions in tgt_actions)
            optimizer.zero_grad()

            batch_loss = -model(src_sents, tgt_nodes, tgt_tokens, tgt_actions).sum()
            loss = batch_loss / num_words_to_predict
            loss.backward()
            optimizer.step()

            cum_loss += batch_loss.item()
            cum_tgt_words += num_words_to_predict

        print('epoch = %d, loss = %.2f, time_elapsed = %.2f'
            % (epoch, cum_loss / cum_tgt_words, time.time() - begin_time))
        #reset epoch progress vars
        cum_loss = .0
        cum_tgt_words = 0

        #perform validation
        dev_loss = validate(model, dev_src, dev_tgt, lang, dev_batch_size)
        is_better = epoch == 0 or dev_loss < min(hist_dev_losses)
        hist_dev_losses.append(dev_loss)
        
        if is_better:
            #reset patience
            patience = 0
            #save model
            #model.module.save(model_save_path)
            model.save(model_save_path)

        else:
            patience += 1
            if patience == int(args['--patience']):
                print('early stopping: dev loss = %.2f' %(dev_loss))
                return

        print('validation: dev loss = %.2f' %(dev_loss))

        #update lr after every 2 epochs
        lr = init_lr / 2 ** (epoch // 2)
        for param_group in optimizer.param_groups:
                param_group['lr'] = lr

def test(args, model):
    """
    test model by generating the target language lf
    @param args (dict): command line args
    @params model
    """
    lang = args['--lang']
    if lang == 'lambda':
        from lang.Lambda.hypothesis import Hypothesis
        from lang.Lambda.parse import ast_to_logical_form

    else:
        print('language:  %s currently not supported' % (lang))

    model = model.to(device)

    test_src, test_tgt = read_corpus(args['TEST_FILE'])
    
    tgt_actions = decode(model, lang, test_src)

    tgt_hyps = []
    for actions in tgt_actions:
        hyp = Hypothesis()
        for action in actions:
            hyp.apply_action(action)
        tgt_hyps.append(hyp)
    pred_tgt = [ast_to_logical_form(hyp.tree).to_string() for hyp in tgt_hyps]

    save_sents(pred_tgt, args['OUTPUT_FILE'])

    em = comp_exact_match(refs=test_tgt, hyps=pred_tgt)
    print('ACC = %.2f' % (em))

if __name__ == "__main__":
    import sys
    args = docopt(__doc__)
    model_type = args['--model-type']
    if args['train']:
        vocab = Vocab.load(args['--vocab'])
        nodes = Node.load(args['--nodes'])
        rules = Rule.load(args['--rules'])

        if model_type == 'TRANS':
            from models.trans_vanilla import TransVanilla
            model = TransVanilla(embed_size=int(args['--embed-size']),
                            hidden_size=int(args['--hidden-size']),
                            vocab=vocab,
                            nodes=nodes,
                            rules=rules,
                            dropout_rate=float(args['--dropout']))
            if args['--save-model-name'] == None:
                args['--save-model-name'] = 'trans.pt'
        elif model_type == 'Seq2AST':
            from models.seq_to_ast import Seq2AST
            model = Seq2AST(embed_size=int(args['--embed-size']),
                            hidden_size=int(args['--hidden-size']),
                            vocab=vocab,
                            nodes=nodes,
                            rules=rules,
                            dropout_rate=float(args['--dropout']))
            if args['--save-model-name'] == None:
                args['--save-model-name'] = 'seq2ast.pt'
        else:
            print('unknown model')
            sys.exit()
        train(args, model)
    elif args['transfer'] or args['test']:
        if model_type == 'TRANS':
            from models.trans_vanilla import TransVanilla
            model = TransVanilla.load(args['MODEL_PATH'])
        elif model_type == 'Seq2AST':
            from models.seq_to_ast import Seq2AST
            model = Seq2AST.load(args['MODEL_PATH'])
        else:
            print('unknown model')
            sys.exit()
        if args['transfer']:
            train(args, model)
        else:
            test(args, model)
