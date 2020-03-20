#!/usr/bin/env python
"""
run script to train and test our neural model

Usage:
    run_transformer.py train --lang=<str> --train-data=<file> --dev-data=<file> --vocab=<file> --nodes=<file> --rules=<file> [options]
    run_transformer.py test MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE [options]

Options:
    -h --help                       show this screen.
    --lang=<str>                    target language
    --train-data=<file>             train data file
    --dev-data=<file>               dev data file
    --vocab=<file>                  vocab file
    --nodes=<file>                  node file
    --rules=<file>                  rule file
    --batch-size=<int>              batch size [default: 8]
    --embed-size=<int>              embedding size [default: 512]
    --hidden-size=<int>             hidden size [default: 2048]
    --max-epoch=<int>               max epoch [default: 20]
    --patience=<int>                num epochs early stopping [default: 5]
    --dropout=<float>               dropout rate [default: 0.1]
    --lr=<float>                    learning rate [default: 1e-4]
    --save-model-name=<file>        save model name [default: trans_vanilla.pt]
    --beam-size=<int>               beam size [default: 4]
    --max-decoding-time-step=<int>  max number of decoding time steps [default: 50]
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
from utils import read_corpus, batch_iter, save_sents
from trans_vanilla import TransVanilla

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
        for src_sents, tgt_nodes, tgt_rules in batch_iter(dev_src, dev_tgt, lang, batch_size):
            num_words_to_predict = sum(len(rules) for rules in tgt_rules)
            loss = -model(src_sents, tgt_nodes, tgt_rules).sum()
            
            cum_loss += loss
            cum_tgt_words += num_words_to_predict
        
        dev_loss = cum_loss / cum_tgt_words
    
    if was_training:
        model.train()

    return dev_loss

def decode(model, test_data_src, beam_size, max_decoding_time_step):
    """
    run inference on model to generate target sentences
    @param model (TransCopy)
    @param test_data_src (list[list[str]): list of test source sentences
    @param beam_size (int): beam size
    @param max_decoding_time_step (int): maximum decoding time steps
    @return gen_tgt_rules (list[list[str]])
    """
    was_training = model.training
    model.eval()

    gen_tgt_rules = []
    with torch.no_grad():
        for src in test_data_src:
            gen_rules = model.beam_search(src, beam_size, max_decoding_time_step)
            gen_tgt_rules.append(gen_rules)

    if was_training:
        model.train
    return gen_tgt_rules

def train(args):
    """
    train our neural model
    @param args (dict): command line args
    """
    lang = args['--lang']

    train_src, train_tgt = read_corpus(args['--train-data'])

    dev_src, dev_tgt = read_corpus(args['--dev-data'])

    train_batch_size = dev_batch_size = int(args['--batch-size'])
    model_save_path = 'models/' + lang + '_' + args['--save-model-name']
    vocab = Vocab.load(args['--vocab'])
    nodes = Node.load(args['--nodes'])
    rules = Rule.load(args['--rules'])

    model = TransVanilla(embed_size=int(args['--embed-size']),
                    hidden_size=int(args['--hidden-size']),
                    vocab=vocab,
                    nodes=nodes,
                    rules=rules,
                    dropout_rate=float(args['--dropout']))
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
        for src_sents, tgt_nodes, tgt_rules in batch_iter(train_src, train_tgt, lang, batch_size=train_batch_size, shuffle=True):
                
            num_words_to_predict = sum(len(tgt_rule) for tgt_rule in tgt_rules)
            optimizer.zero_grad()

            batch_loss = -model(src_sents, tgt_nodes, tgt_rules).sum()
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

def test(args):
    """
    test TransCopy model by generating target sentences
    @param args (dict): command line args
    """
    model = TransCopy.load(args['MODEL_PATH'])
    model = model.to(device)

    (test_data_src, test_data_tgt), sent_str_map = read_corpus(args['TEST_SOURCE_FILE'], args['TEST_TARGET_FILE'])
    
    gen_tgt_rules = decode(model, test_data_src, 
                            beam_size=int(args['--beam-size']),
                            max_decoding_time_step=int(args['--max-decoding-time-step']))

    gen_tgt_sents = [rules_to_code(gen_rules, code) for (gen_rules, code) in zip(gen_tgt_rules, test_data_tgt)]

    save_sents(gen_tgt_sents, args['OUTPUT_FILE'])

    bleu_score = compute_bleu_score(refs=test_data_tgt, hyps=gen_tgt_sents)
    #em = compute_exact_match(refs=test_data_tgt, hyps=gen_tgt_sents)
    print('BLEU score = %.2f' % (bleu_score))

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['train']:
        train(args)
    elif args['test']:
        test(args)
