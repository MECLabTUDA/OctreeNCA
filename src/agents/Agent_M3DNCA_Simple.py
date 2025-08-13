import torch
from src.agents.Agent_UNet import UNetAgent
import torch.nn.functional as F
import random
from src.agents.Agent_MedSeg3D import Agent_MedSeg3D
from src.utils.MyDataParallel import MyDataParallel

class M3DNCAAgent(UNetAgent):
    """Base agent for training UNet models
    """
    def initialize(self):
        super().initialize()

        if self.exp.get_from_config('performance.data_parallel'):
            self.model = MyDataParallel(self.model)

        self.model.to(self.device)


    def get_outputs(self, data: tuple, full_img=True, **kwargs) -> dict:
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        inputs, targets = data['image'], data['label']

        #inputs: BCHWD
        #targets: BHWDC

        inputs = inputs.permute(0, 2, 3, 4, 1)

        out = self.model(inputs, targets, self.exp.get_from_config('trainer.batch_duplication'))
        out["target_unpatched"] = targets
        return out 

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
        rnd = random.randint(0, 1000000000)
        random.seed(rnd)
        out = self.get_outputs(data)
        #print("______________________")
        if self.exp.get_from_config('trainer.train_quality_control') in ["NQM", "MSE"]:
            raise NotImplementedError("NQM and MSE not implemented for 3D")
            random.seed(rnd)
            outputs2, targets2 = self.get_outputs(data)
        loss = 0
        loss_ret = {}
        #print(outputs.shape)
        #print(targets.shape)
        #exit()
        loss, loss_ret = loss_f(**out)

        # CALC NQM
        if self.exp.get_from_config('trainer.train_quality_control') == "NQM":
            stack = torch.stack([outputs, outputs2], dim=0)
            outputs = torch.sigmoid(torch.mean(stack, dim=0))
            stack = torch.sigmoid(stack)
            if torch.sum(stack) != 0:
                mean = torch.sum(stack, axis=0) / stack.shape[0]
                stdd = torch.zeros(mean.shape).to(self.device)
                for id in range(stack.shape[0]):
                    img = stack[id] - mean
                    img = torch.pow(img, 2)
                    stdd = stdd + img
                stdd = stdd / stack.shape[0]
                stdd = torch.sqrt(stdd)

                print("STDD", torch.min(stdd), torch.max(stdd), torch.sum(outputs))

                if torch.min(stdd) > 0:
                    nqm = torch.sum(stdd) / torch.sum(outputs)

                    if nqm > 0:
                        print("NQM: ", nqm)
                        loss = loss + nqm #
        elif self.exp.get_from_config('trainer.train_quality_control') == "MSE":
            loss += F.mse_loss(outputs, outputs2)

            #print(nqm)

        if loss != 0:
            loss.backward()

            if gradient_norm or self.exp.get_from_config('experiment.logging.track_gradient_norm'):
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

            if gradient_norm:
                print("GRADIENT NORM")
                max_norm = 1.0
                # Gradient normalization

                # Calculate scaling factor and scale gradients if necessary
                scale_factor = max_norm / (total_norm + 1e-6)
                if scale_factor < 1:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(scale_factor)
            
            if self.exp.get_from_config('experiment.logging.track_gradient_norm'):
                if not hasattr(self, 'epoch_grad_norm'):
                    self.epoch_grad_norm = []
                self.epoch_grad_norm.append(total_norm)

            self.optimizer.step()
            if not self.exp.get_from_config('trainer.update_lr_per_epoch'):
                self.update_lr()
        return loss_ret
    