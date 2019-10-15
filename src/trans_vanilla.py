from __future__ import division
import math, copy

from torch_dependencies import *
from trans_encoder import TransEncoder
from model_embeddings import ModelEmbeddings

class TransVanilla(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate):
        super(TransVanilla, self).__init__()
        self.embeddings = ModelEmbeddings(embed_size, vocab)
        self.d_model = embed_size
        self.d_ff = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        
        self.trans_encoder = TransEncoder(self.d_model, self.d_ff, self.dropout_rate)
        self.encoder_blocks = self.clone(self.trans_encoder, n=6)

    def forward(self, src):
        """
        src: (list[list[str]])
        """
        src_lens = [len(s) for s in src]
        src_mask = torch.ones(len(src), max(src_lens), max(src_lens), dtype=torch.long).to(self.device)
        for i, src_len in enumerate(src_lens):
            src_mask[i, src_len:, src_len:] = 0

        src_padded = self.vocab.src.sents2Tensor(src).to(self.device)
        encoded_src = self.encode(src_padded, src_mask)
        return encoded_src


    def encode(self, src, src_mask=None):
        x = self.embeddings.src_embedding(src)
        for encoder in self.encoder_blocks:
            x = encoder(x, src_mask)
        return x

    #move clone -> utils
    def clone(self, block, n):
        return nn.ModuleList([copy.deepcopy(block) for _ in range(n)])

    @property
    def device(self):
        """
        property decorator for device
        """
        return self.embeddings.src_embedding.weight.device
