import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.in_permute = Rearrange('b c h w -> b h w c')
        self.out_permute = Rearrange('b h w c -> b c h w')
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.out_permute(self.norm(self.in_permute(x)))