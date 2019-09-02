#!/usr/bin/env python
"""
run script to train and test our neural model

Usage:
    run.py train --train-src=<file> --train-tgt=<file> --vocab=<file> [options]

Options:
    -h --help                       show this screen.
    --train-src=<file>              train source file
    --train-tgt=<file>              train target file
    --vocab=<file>                  vocab file
    --batch-size=<int>              batch size [default: 32]
    --embed-size=<int>              embedding size [default: 128]
    --hiden-size=<int>              hidden size [default: 256]
    --clip-grad=<float>             gradient clipping [default: 5.0]
    --max-epoch=<int>               max epoch [default: 15]
    --patientce=<int>               num epochs early stopping [default: 5]
    --lr=<float>                    learning rate [default: 1e-3]
    --lr-decay=<float>              learning rate decay [default: 0.5]
    --dropout=<float>               dropout rate [default: 0.3]
    --save-model-to=<file>          save model path [dafault: model.pt] 
"""
from __future__ import division

import time
import math
from docopt import docopt

import torch
import torch.nn.functional as F

from vocab import Vocab
from utils import read_corpus, batch_iter
from seq_to_seq import Seq2Seq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    """
    train our neural model
    @param args (dict): command line args
    """
    train_data_src = read_corpus(args['--train-src'], domain='src')
    train_data_tgt = read_corpus(args['--train-tgt'], domain='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    model_save_path = args['--save-model-to']

    vocab = Vocab.load(args['--vocab'])

    model = Seq2Seq(embed_size=int(args['--embed-size']),
                    hidden_size=int(args['--hidden-size']),
                    dropout_rate=float(args['--dropout']),
                    vocab=vocab)
    model.train()
    model = model.to(device)

    init_lr = float(args['--lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    total_loss = .0
    total_tgt_words = 0
    begin_time = time.time()
    for epoch in range(int(args['--max-epoch'])):
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            num_words_to_predict = sum(len(tgt_sent[1:]) for tgt_sent in tgt_sents)

            optimizer.zero_grad()

            batch_loss = -model(src_sents, tgt_sents).sum()
            loss = batch_loss / num_words_to_predict
            
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            total_loss = batch_loss.item()
            total_tgt_words += num_words_to_predict

        print('epoch = %d, loss = %.2f, time_elapsed = %.2f'
            % (epoch, total_loss / total_tgt_words, time.time() - begin_time))
        #reset epoch progress vars
        total_loss = .0
        total_tgt_words = 0

        #update lr after every 2 epochs
        lr = init_lr / 2 ** (epoch // 2)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr       

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['train']:
        train(args)
