import einops
import torch
import torch.nn as nn
import src.losses.DiceBCELoss
import src.losses.OverflowLoss
import src.losses.MaskedL1Loss
import src.losses.IntermediateSupervision
import src.losses.BCELoss
import src.losses.DiceLoss


class WeightedLosses(nn.Module):
    def __init__(self, config):
        super(WeightedLosses, self).__init__()
        assert len(config['trainer.losses']) == len(config['trainer.loss_weights']), f"{config['trainer.losses']} and {config['trainer.loss_weights']} must have the same length"
        self.losses = []
        self.weights = []
        for i, _ in enumerate(config['trainer.losses']):
            try:
                self.losses.append(eval(config['trainer.losses'][i])(config=config))
            except TypeError:
                try:
                    loss_parameters = config['trainer.losses.parameters'][i]
                    self.losses.append(eval(config['trainer.losses'][i])(**loss_parameters))
                except TypeError:
                    self.losses.append(eval(config['trainer.losses'][i])())
            
            self.weights.append(config['trainer.loss_weights'][i])
            
            

    def forward(self, **kwargs):
        loss = 0
        loss_ret = {}
        for i, _ in enumerate(self.losses):
            try:
                r = self.losses[i](**kwargs)
            except TypeError as e:
                input(f"TypeError {e}")
                if kwargs["logits"].dim() == 5:
                    logits = einops.rearrange(kwargs["logits"], "b h w d c -> b c h w d")
                    target = einops.rearrange(kwargs["target"], "b h w d c -> b c h w d")
                else:
                    assert kwargs["logits"].dim() == 4, f"Expected 4D tensor, got {kwargs['logits'].dim()}"
                    logits = einops.rearrange(kwargs["logits"], "b h w c -> b c h w")
                    target = einops.rearrange(kwargs["target"], "b h w c -> b c h w")
                r = self.losses[i](logits, target)
            
            if isinstance(r, tuple):
                l, d = r
            else:
                l = r
                d = {'loss': l.item()}
            loss += l * self.weights[i]
            for k, v in d.items():
                loss_ret[f"{self.losses[i].__class__.__name__}/{k}"] = d[k] * self.weights[i]
        return loss, loss_ret