import torch
from einops import rearrange
from .vitca_utils import neighbourhood_filters
from scipy import signal
import numpy as np

def clamp(x, min_val, max_val):
    return max(min(x, max_val), min_val)

def neighbourhood_filters_3d(neighbourhood_size, device):
    height, width, depth = neighbourhood_size
    impulses = []
    for i in range(height):
        for j in range(width):
            for k in range(depth):
                impulse = signal.unit_impulse((height, width, depth), idx=(i,j,k), dtype=np.float32)
                impulses.append(impulse)
    filters = torch.tensor(np.stack(impulses), device=device)
    return filters

class LocalizeAttention3D(torch.nn.Module):
    def __init__(self, attn_neighbourhood_size, device) -> None:
        super().__init__()
        self.attn_neighbourhood_size = attn_neighbourhood_size
        self.device = device
        self.attn_filters = neighbourhood_filters_3d(self.attn_neighbourhood_size, self.device)

    def forward(self, x, height, width, depth):
        '''attn_filters: [filter_n, h, w]'''
        b, h, _, d = x.shape
        y = rearrange(x, 'b h (i j k) d -> (b h d) 1 i j k', i=height, j=width, k=depth)
        y = torch.nn.functional.conv3d(y, self.attn_filters[:, None], padding='same')
        _x = rearrange(y, '(b h d) filter_n i j k -> b h (i j k) filter_n d', b=b, h=h, d=d)
        return _x