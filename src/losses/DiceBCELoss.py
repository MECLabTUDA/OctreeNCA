import torch


class DiceBCELoss(torch.nn.Module):
    r"""Dice BCE Loss
    """
    def __init__(self, useSigmoid: bool = True, smooth: float = 1) -> None:
        r"""Initialisation method of DiceBCELoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        self.smooth = smooth
        super(DiceBCELoss, self).__init__()

    def compute(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.useSigmoid:
            output = torch.sigmoid(output)
        output = torch.flatten(output) 
        target = torch.flatten(target)
        
        intersection = (output * target).sum()                            
        dice_loss = 1 - (2.*intersection + self.smooth)/(output.sum() + target.sum() + self.smooth)  
        BCE = torch.nn.functional.binary_cross_entropy(output, target, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs):
        loss_ret = {}
        loss = 0
        if len(logits.shape) == 5 and target.shape[-1] == 1:
            for m in range(target.shape[-1]):
                loss_loc = self.compute(logits[..., m], target[...])
                loss = loss + loss_loc
                loss_ret[f"mask_{m}"] = loss_loc.item()
        else:
            for m in range(target.shape[-1]):
                if 1 in target[..., m]:
                    loss_loc = self.compute(logits[..., m], target[..., m])
                    loss = loss + loss_loc
                    loss_ret[f"mask_{m}"] = loss_loc.item()

        return loss, loss_ret