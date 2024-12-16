
import torch
import torch.nn as nn
from utils import clones
from residual import LayerNorm, SublayerConnection

class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)


    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn"
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
    

class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        The 1st sublayer: multi-head self-attention mechanism.
        The 2nd sublayer: position-wise fully connected feed-forward network.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        """
        Follow Figure 1 (left) for connections
        """
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayers[1](x, self.feed_forward)
