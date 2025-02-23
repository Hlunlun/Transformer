import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn


from multi_head_attention import MultiHeadedAttention
from positional_encoding import PositionEncoding
from positionwise_feed_forward import PositionwiseFeedForward
from encoder_decoder import EncoderDecoder
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from embeddings import Embeddings
from generator import Generator

seaborn.set_context(context="talk")


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Healper: Construct a model from hyperparameters"""

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # This was important from their code
    # Initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# Small example model.
tmp_model = make_model(10, 10, 2)
print(tmp_model)
