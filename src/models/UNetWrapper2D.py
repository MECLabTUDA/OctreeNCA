import torch, einops
import torch.nn as nn
from unet import UNet
import matplotlib.pyplot as plt

class UNetWrapper2D(nn.Module):
    def __init__(self, model: UNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, y: torch.Tensor=None, batch_duplication: int=1):
        if y is not None:
            y = einops.rearrange(y, 'b c h w -> b h w c')
        if self.training:
            # training
            if batch_duplication != 1:
                x = einops.repeat(x, 'b c h w -> (b dup) c d h w', dup=batch_duplication)
                y = einops.repeat(y, 'b h w c -> (b dup) h w c', dup=batch_duplication)
            seg = self.model(x)
            seg = einops.rearrange(seg, 'b c h w -> b h w c')
            return {"logits":seg, "target": y}
        else:
            # evaluation
            seg = self.model(x)
            seg = einops.rearrange(seg, 'b c h w -> b h w c')
            return {"logits":seg}