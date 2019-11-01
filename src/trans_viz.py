#!/usr/bin/env python

#written in notebook style
import matplotlib.pyplot as plt
import seaborn
import torch
from os import sys

from utils import read_corpus, subsequent_mask
from trans_vanilla import TransVanilla

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def draw(data, x, y, ax):
    seaborn.heatmap(data.cpu().numpy(), 
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                    cbar=False, ax=ax)

if __name__ == "__main__":
    model = TransVanilla.load(sys.argv[1])
    model = model.to(device)
    src_sents = read_corpus('../data/selected_src', domain='src')
    tgt_sents = read_corpus('../data/selected_tgt', domain='tgt')
    assert len(src_sents) == len(tgt_sents)

    model.eval()
    num_blocks = 6
    with torch.no_grad():
        for sent_num, (src_sent, tgt_sent) in enumerate(zip(src_sents[:5], tgt_sents[:5])):
            src, tgt = [src_sent], [tgt_sent]
            src = model.vocab.src.sents2Tensor(src).to(device)
            tgt = model.vocab.tgt.sents2Tensor(tgt).to(device)

            x = model.pe(model.embeddings.src_embedding(src))
            for layer, encoder in enumerate(model.encoder_blocks):
                x, q_key_dots = encoder(x)
                if layer < num_blocks - 1 and layer % 2 != 0: continue 
                fig, axis = plt.subplots(1, len(q_key_dots), figsize=(20, 10))
                for i, q_key_dot in enumerate(q_key_dots):
                    draw(q_key_dot.squeeze(0).data, 
                        src_sent, src_sent if i ==0 else [], ax=axis[i])
                fig.savefig('../results/ex_' + str(sent_num+1) + '_encoder_layer_' + str(layer+1) + '.eps')

            y = model.pe(model.embeddings.tgt_embedding(tgt))
            for layer, decoder in enumerate(model.decoder_blocks):
                y, q_key_dots, q_key_mask_dots = decoder(x, y)
                if layer < num_blocks - 1 and layer % 2 != 0: continue 
                fig1, axis1 = plt.subplots(1, len(q_key_dots), figsize=(20, 10))
                fig2, axis2 = plt.subplots(1, len(q_key_mask_dots), figsize=(20, 10))
                for i, (q_key_dot, q_key_mask_dot) in enumerate(zip(q_key_dots, q_key_mask_dots)):
                    draw(q_key_dot.squeeze(0).data, 
                        tgt_sent, tgt_sent if i ==0 else [], ax=axis1[i])
                    draw(q_key_mask_dot.squeeze(0).data, 
                        x=src_sent, y=tgt_sent if i ==0 else [], ax=axis2[i])
                fig1.savefig('../results/ex_' + str(sent_num+1) + '_decoder_layer_' + str(layer+1) + '.eps')
                fig2.savefig('../results/ex_' + str(sent_num+1) + '_enc_dec_layer_' + str(layer+1) + '.eps')
