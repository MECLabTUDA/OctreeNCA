import torch
import torch.nn as nn
import numpy as np
import src.losses.DiceBCELoss
import matplotlib.pyplot as plt
import torch.nn.functional as F
import einops
import matplotlib.pyplot as plt

class DiceBCELossInterSuperv(nn.Module):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.dice_bce_loss = src.losses.DiceBCELoss.DiceBCELoss()
        self.octree_res = [r[0] for r in config["model.octree.res_and_steps"]]
        self.octree_res = self.octree_res[::-1]

    def compute_factors(self, num_levels: int):
        #copied from nnUNet
        weights = np.array([1 / (2 ** i) for i in range(num_levels)])
        mask = np.array([True] + [True if i < num_levels - 1 else False for i in range(1, num_levels)])
        weights[~mask] = 0
        weights = weights / weights.sum()
        self.weights = np.flip(weights)

    def downsample(self, tensor: torch.Tensor, size)-> torch.Tensor:
        tensor = einops.rearrange(tensor, "b h w d c -> b c h w d")
        tensor = F.interpolate(tensor, size=size, mode="nearest")
        return einops.rearrange(tensor, "b c h w d -> b h w d c")

    def forward(self,
                intermediate_outputs: list[torch.Tensor], 
                target_unpatched: torch.Tensor, 
                intermediate_patches: list[np.ndarray], **kwargs) -> tuple:
        assert len(intermediate_outputs) == len(intermediate_patches)
        # target_unpatched.shape: BHWDC
        # intermediate_outputs[i].shape: BHWDC
        target_unpatched.names = None



        if not hasattr(self, "weights"):
            self.compute_factors(len(intermediate_outputs))

        
        loss_sum = 0
        loss_dict = {}

        for level in range(len(intermediate_outputs)):
            if self.weights[level] <= 0.001:
                continue
            current_patch = intermediate_patches[level]
            patched_target = torch.zeros_like(intermediate_outputs[level])
            target_unpatched_downsampled = self.downsample(target_unpatched, self.octree_res[level])

            for b in range(intermediate_patches[level].shape[0]):
                patched_target_b = target_unpatched_downsampled[b,
                                                    current_patch[b,0,0]:current_patch[b,1,0],
                                                    current_patch[b,0,1]:current_patch[b,1,1],
                                                    current_patch[b,0,2]:current_patch[b,1,2]]
                assert patched_target_b.shape == intermediate_outputs[level][b].shape, f"{patched_target_b.shape} != {intermediate_outputs[level][b].shape}"
                patched_target[b] = patched_target_b

            loss, loss_pred = self.dice_bce_loss(output=intermediate_outputs[level], 
                                                 target=patched_target)


            for key in loss_pred.keys():
                loss_dict[f"scale_{level}/{key}"] = loss_pred[key]

            loss_sum = loss_sum + self.weights[level] * loss

        return loss_sum, loss_dict