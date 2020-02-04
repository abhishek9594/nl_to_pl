#!/usr/bin/env python
"""
run script to train and test our neural model

Usage:
    run_transformer.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    run_transformer.py test MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE [options]

Options:
    -h --help                       show this screen.
    --train-src=<file>              train source file
    --train-tgt=<file>              train target file
    --dev-src=<file>                dev source file
    --dev-tgt=<file>                dev target file
    --vocab=<file>                  vocab file
    --batch-size=<int>              batch size [default: 16]
    --embed-size=<int>              embedding size [default: 512]
    --hidden-size=<int>             hidden size [default: 2048]
    --max-epoch=<int>               max epoch [default: 20]
    --patience=<int>                num epochs early stopping [default: 5]
    --dropout=<float>               dropout rate [default: 0.1]
    --lr=<float>                    learning rate [default: 1e-4]
    --save-model-to=<file>          save model path [default: trans_copy.pt]
    --beam-size=<int>               beam size [default: 4]
    --max-decoding-time-step=<int>  max number of decoding time steps [default: 50]
"""
from __future__ import division

import time, math
from docopt import docopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import Vocab
from nn.utils import read_corpus, batch_iter, save_sents, compute_bleu_score, compute_exact_match
from nn.trans_copy import TransCopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(model, dev_data, batch_size=32):
    """
    validate model on dev set
    @param model (TransCopy)
    @param dev_data (list[(list[str], list[str])]): list of tuples of source and target sentences
    @param batch_size (int): batch size
    @return dev_loss (float): cross entropy loss on dev set
    """
    was_training = model.training
    model.eval()
    
    cum_loss = .0
    cum_tgt_words = 0

    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            num_words_to_predict = sum(len(tgt_sent[1:]) for tgt_sent in tgt_sents)
            loss = -model(src_sents, tgt_sents).sum()
            
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
    @return gen_tgt_sents (list[list[str]])
    """
    was_training = model.training
    model.eval()

    gen_tgt_sents = []
    with torch.no_grad():
        for src in test_data_src:
            gen_tgt_sent = model.beam_search(src, beam_size, max_decoding_time_step)
            gen_tgt_sents.append(gen_tgt_sent)

    if was_training:
        model.train
    return gen_tgt_sents

def train(args):
    """
    train our neural model
    @param args (dict): command line args
    """
    train_data_src = read_corpus(args['--train-src'], domain='src')
    train_data_tgt = read_corpus(args['--train-tgt'], domain='tgt')

    dev_data_src = read_corpus(args['--dev-src'], domain='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], domain='tgt')
    
    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = dev_batch_size = int(args['--batch-size'])
    model_save_path = args['--save-model-to']
    vocab = Vocab.load(args['--vocab'])

    model = TransCopy(embed_size=int(args['--embed-size']),
                    hidden_size=int(args['--hidden-size']),
                    vocab=vocab,
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
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            num_words_to_predict = sum(len(tgt_sent[1:]) for tgt_sent in tgt_sents)
            optimizer.zero_grad()

            batch_loss = -model(src_sents, tgt_sents).sum()
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
        dev_loss = validate(model, dev_data, dev_batch_size)
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

    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], domain='src')
    test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], domain='tgt')
    
    gen_tgt_sents = decode(model, test_data_src, 
                            beam_size=int(args['--beam-size']),
                            max_decoding_time_step=int(args['--max-decoding-time-step']))

    save_sents(gen_tgt_sents, args['OUTPUT_FILE'])

    bleu_score = compute_bleu_score(refs=test_data_tgt, hyps=gen_tgt_sents)
    em = compute_exact_match(refs=test_data_tgt, hyps=gen_tgt_sents)
    print('BLEU score = %.2f, EM = %.2f' % (bleu_score, em))

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['train']:
        train(args)
    elif args['test']:
        test(args)
