import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """Define standard linear + softmax generation step"""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def foward(self, x):
        """
        last layer that filter the higher probability word
        """
        return F.log_softmax(self.proj(x), dim=-1)
    