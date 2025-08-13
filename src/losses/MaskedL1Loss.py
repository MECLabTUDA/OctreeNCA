import torch, torch.nn as nn
import torch.nn.functional as F

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, **kwargs) -> tuple:
        mask = kwargs['loss_mask']
        loss = F.l1_loss(pred, target, reduction="none")
        loss = loss * mask
        loss = torch.sum(loss) / (torch.sum(mask) + 1e-5)
        return loss, {"loss": loss.item()}