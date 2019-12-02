#!/usr/bin/env python

#written in notebook style
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn
import torch
from os import sys

from utils import read_corpus, subsequent_mask
from trans_copy import TransCopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def draw(data, x, y, ax):
    seaborn.heatmap(data.cpu().numpy(), 
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                    cbar=False, ax=ax)

if __name__ == "__main__":
    model = TransCopy.load(sys.argv[1])
    model = model.to(device)
    src_lang = sys.argv[2]
    tgt_lang = sys.argv[3]
    src_sents = read_corpus(sys.argv[4], domain='src')
    tgt_sents = read_corpus(sys.argv[5], domain='tgt')
    assert len(src_sents) == len(tgt_sents)

    model.eval()
    num_blocks = 6
    with torch.no_grad():
        for sent_num, (src_sent, tgt_sent) in enumerate(zip(src_sents, tgt_sents)):
            src, tgt = [src_sent], [tgt_sent]
            src = model.vocab.src.sents2Tensor(src).to(device)
            tgt = model.vocab.tgt.sents2Tensor(tgt).to(device)

            #run encoder to obtain (k, v)
            x = model.pe(model.embeddings.src_embedding(src))
            for layer, encoder in enumerate(model.encoder_blocks):
                x, _ = encoder(x)

            #run decoder to obtain the Q-K attention heat map
            y = model.pe(model.embeddings.tgt_embedding(tgt))
            for layer, decoder in enumerate(model.decoder_blocks):
                y, _, q_key_src_dots = decoder(x, y)

                #only draw last layer heatmap
                if layer < num_blocks - 1: continue 
                fig, axis = plt.subplots(1, len(q_key_src_dots), figsize=(20, 10))
                for i, q_key_src_dot in enumerate(q_key_src_dots):
                    draw(q_key_src_dot.squeeze(0).data, 
                        x=src_sent, y=tgt_sent if i ==0 else [], ax=axis[i])
                fig.savefig('../results/' + src_lang + '_' + tgt_lang + '_' + 'ex_' + str(sent_num+1) + '_' + str(layer+1) + '.eps')
