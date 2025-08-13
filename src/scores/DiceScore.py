import torch
import matplotlib.pyplot as plt

class DiceScore(torch.nn.Module):


    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        #target: BHWDC or BHWC
        #output: BHWDC or BHWC
        output = output > 0
        d = {}

        for m in range(target.shape[-1]):
            if 1 in target[..., m]:
                dice = torch.sum(output[..., m] * target[..., m]) * 2.0 / (torch.sum(output[..., m]) + torch.sum(target[..., m]))
                d[m] = dice.item()
            else:
                if not 1 in output[..., m]:
                    continue
                else:
                    d[m] = 0

        return d