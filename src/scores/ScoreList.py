import torch
import torch.nn as nn
import src.scores.DiceScore
import src.scores.IoUScore
import src.scores.PatchwiseDiceScore
import src.scores.PatchwiseIoUScore

class ScoreList(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scores = []
        for i, _ in enumerate(config['experiment.task.score']):
            self.scores.append(eval(config['experiment.task.score'][i])())
            

    def forward(self, pred, target, **kwargs) -> dict:
        #target: BHWDC or BHWC
        #output: BHWDC or BHWC
        loss_ret = {}
        for i, _ in enumerate(self.scores):
            try:
                s = self.scores[i](pred, target, **kwargs)
            except TypeError:
                s = self.scores[i](pred, target)
            if isinstance(s, dict):
                for k, v in s.items():
                    loss_ret[f"{self.scores[i].__class__.__name__}/{k}"] = s[k]
            else:
                loss_ret[f"{self.scores[i].__class__.__name__}/score"] = s.item()
        return loss_ret