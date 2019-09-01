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
import pickle
from docopt import docopt

import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
#TODO

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['train']:
        train(args)
