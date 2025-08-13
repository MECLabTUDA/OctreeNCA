import torch
import matplotlib.pyplot as plt

class PatchwiseDiceScore(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor, patient_id: str, **kwargs):
        #target: BHWDC or BHWC
        #output: BHWDC or BHWC

        if not hasattr(self, 'true_positives'):
            self.true_positives = []
            for m in range(target.shape[-1]):
                self.true_positives.append(dict())

        if not hasattr(self, 'false_positives'):
            self.false_positives = []
            for m in range(target.shape[-1]):
                self.false_positives.append(dict())

        if not hasattr(self, 'false_negatives'):
            self.false_negatives = []
            for m in range(target.shape[-1]):
                self.false_negatives.append(dict())

        
        output = (output > 0).int()
        d = {}
        for m in range(target.shape[-1]):
            if not patient_id in self.true_positives[m]:
                self.true_positives[m][patient_id] = 0
                self.false_positives[m][patient_id] = 0
                self.false_negatives[m][patient_id] = 0

            self.true_positives[m][patient_id] += torch.sum(output[..., m] * target[..., m]).item()

            self.false_positives[m][patient_id] += torch.sum(output[..., m] * (1 - target[..., m])).item()

            self.false_negatives[m][patient_id] += torch.sum((1 - output[..., m]) * target[..., m]).item()

            d[m] = 2.0 * self.true_positives[m][patient_id] / (2.0 * self.true_positives[m][patient_id] + self.false_positives[m][patient_id] + self.false_negatives[m][patient_id] + 0.00001)


        #plt.subplot(1, 2, 1)
        #plt.imshow(output[0, ..., 0].cpu())
        #plt.subplot(1, 2, 2)
        #plt.imshow(target[0, ..., 0].cpu())
        #plt.show()
        
        return d