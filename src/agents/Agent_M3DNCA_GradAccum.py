import torch
from torch.nn.modules import Module
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
import torch.nn.functional as F
import random

class M3DNCAAgentGradientAccum(M3DNCAAgent):
    """Base agent for training UNet models
    """

    def __init__(self, model: Module):
        super().__init__(model)
        assert False, "do not use, something here is wrong!" #The loss goes down, but the dice score in the end is very low

    def compute_gradients(self, inputs: torch.Tensor, full_targets: torch.Tensor, loss_f: torch.nn.Module,
                          norm_coeff: float) -> dict:
        rnd = random.randint(0, 1000000000)
        random.seed(rnd)
        outputs, targets = self.model(inputs, full_targets)
        #print("______________________")
        random.seed(rnd)
        outputs2, targets2 = self.model(inputs, full_targets)
        del targets2

        loss = 0
        loss_ret = {}
        if len(outputs.shape) == 5 and targets.shape[-1] == 1:
            for m in range(targets.shape[-1]):
                loss_loc = norm_coeff * loss_f(outputs[..., m], targets[...])
                loss = loss + loss_loc
                loss_ret[m] = loss_loc.item()
        else:
            for m in range(targets.shape[-1]):
                if 1 in targets[..., m]:
                    loss_loc = norm_coeff * loss_f(outputs[..., m], targets[..., m])
                    loss = loss + loss_loc
                    loss_ret[m] = loss_loc.item()

        loss += F.mse_loss(outputs, outputs2) * norm_coeff
        if loss != 0:
            loss.backward()
        return loss_ret
        

    def batch_step(self, data: tuple, loss_f: torch.nn.Module, gradient_norm: bool = False) -> dict:
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """
        self.optimizer.zero_grad()
        data = self.prepare_data(data)
        inputs, targets = data['image'], data['label']
        inputs = inputs.permute(0, 2, 3, 4, 1)

        if self.exp.get_from_config('trainer.gradient_accumulation'):#TODO also split across batch duplication
            loss_ret = {}
            for b in range(inputs.shape[0]):
                loss_ret_temp = self.compute_gradients(inputs[b:b+1], targets[b:b+1], loss_f, 1/inputs.shape[0])
                for key in loss_ret_temp:
                    if key not in loss_ret:
                        loss_ret[key] = 0
                    loss_ret[key] += loss_ret_temp[key]
        else:
            loss_ret = self.compute_gradients(inputs, targets, loss_f)


        if gradient_norm:
            print("GRADIENT NORM")
            max_norm = 1.0
            # Gradient normalization
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # Calculate scaling factor and scale gradients if necessary
            scale_factor = max_norm / (total_norm + 1e-6)
            if scale_factor < 1:
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(scale_factor)


        self.optimizer.step()
        if not self.exp.get_from_config('trainer.update_lr_per_epoch'):
            self.update_lr()
        return loss_ret
    