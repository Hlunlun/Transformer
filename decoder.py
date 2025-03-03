import torch.nn as nn
from utils import clones
from residual import LayerNorm, SublayerConnection


# stack of N=6 identical layers
class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layer = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        
        return self.norm(x)
    

class DecoderLayer(nn.Module):
    "Decoder is made of slef-attn, src-attn(from encoder), and feed forward"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        # first sublayer: masked multi-head attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # second sublayer: src_attn -> multi-head attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # third sublayer: feed foward
        return self.sublayer[2](x, self.feed_forward)
    
