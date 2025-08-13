import torch
import matplotlib.pyplot as plt

class IoUScore(torch.nn.Module):


    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        #target: BHWDC or BHWC
        #output: BHWDC or BHWC
        output = output > 0
        d = {}

        for m in range(target.shape[-1]):
            if 1 in target[..., m]:
                iou = torch.sum(output[..., m] * target[..., m]) / (torch.sum(output[..., m]) + torch.sum(target[..., m]) - torch.sum(output[..., m] * target[..., m]))
                d[m] = iou.item()
            else:
                if not 1 in output[..., m]:
                    continue
                else:
                    d[m] = 0

        return d