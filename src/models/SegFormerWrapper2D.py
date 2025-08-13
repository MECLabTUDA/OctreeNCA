import torch, einops
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation
from transformers.modeling_outputs import SemanticSegmenterOutput
import torch.nn.functional as F

class SegFormerWrapper2D(nn.Module):
    def __init__(self, model: SegformerForSemanticSegmentation):
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
            out: SemanticSegmenterOutput = self.model(x)
            seg = out.logits[:, 0:1]
            seg = F.interpolate(seg, size=(y.shape[1], y.shape[2]), mode='nearest')
            seg = einops.rearrange(seg, 'b c h w -> b h w c')
            return {"logits":seg, "target": y}
        else:
            # evaluation
            out: SemanticSegmenterOutput = self.model(x)
            seg = out.logits[:, 0:1]
            seg = F.interpolate(seg, size=(y.shape[1], y.shape[2]), mode='nearest')
            seg = einops.rearrange(seg, 'b c h w -> b h w c')
            return {"logits":seg}