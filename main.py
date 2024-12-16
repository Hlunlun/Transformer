import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F



def subsequent_mask(size):
    """
    Mask out subsequent positions.
    
    Ensure that model can only see left side and cannot see right side token. -- UPDATE (Lun) 2024-12-15
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)==0



if __name__ == '__main__':
    plt.figure(figsize=(5,5))
    plt.imshow(subsequent_mask(20)[0])
    plt.savefig('image/mask.jpg')

   