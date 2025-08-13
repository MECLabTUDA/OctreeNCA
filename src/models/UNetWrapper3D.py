import torch, einops
import torch.nn as nn
from unet import UNet

class UNetWrapper3D(nn.Module):
    def __init__(self, model: UNet):
        super(UNetWrapper3D, self).__init__()
        self.model = model
        #print(model)
        #input("Press Enter to continue...")

    def forward(self, x, y: torch.Tensor=None, batch_duplication: int=1):
        x = einops.rearrange(x, 'b d h w c -> b c d h w')
        if self.training:
            # training
            if batch_duplication != 1:
                x = einops.repeat(x, 'b c d h w -> (b dup) c d h w', dup=batch_duplication)
                y = einops.repeat(y, 'b d h w c -> (b dup) d h w c', dup=batch_duplication)

            out = self.model(x)
            out = einops.rearrange(out, 'b c d h w -> b d h w c')
            return {"logits":out, "target": y}
        else:
            # evaluation
            out = self.model(x)
            out = einops.rearrange(out, 'b c d h w -> b d h w c')
            return {"logits":out}