from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from src.models.Model_BasicNCA3D import BasicNCA3D
import torchio as tio
import random
import math
import torch.nn.functional as F
import subprocess as sp

from src.models.Model_OctreeNCA_3D import OctreeNCA3D

class OctreeNCA3DPatch(OctreeNCA3D):
    r"""Implementation of M3D-NCA
    """
    def __init__(self, channel_n, fire_rate, device, steps=64, hidden_size=128, input_channels=1, output_channels=1, 
                 scale_factor=None, levels=None, kernel_size=None,
                 octree_res_and_steps: list=None, separate_models: bool=False,
                 compile: bool=False,
                 patch_sizes=None):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
        """
        super(OctreeNCA3DPatch, self).__init__(channel_n, fire_rate, device, steps, hidden_size, input_channels, output_channels, scale_factor, levels, kernel_size, octree_res_and_steps, separate_models, compile)


        self.computed_upsampling_scales = []
        for i in range(len(self.octree_res)-1):
            t = []
            for c in range(3):
                t.append(self.octree_res[i][c]//self.octree_res[i+1][c])
            self.computed_upsampling_scales.append(np.array(t).reshape(1, 3))

        self.patch_sizes = patch_sizes

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, batch_duplication=1):
        #x: BHWDC
        #y: BHWDC

        if y is not None:
            y = y.to(self.device)

        if self.training:
            if batch_duplication != 1:
                x = torch.cat([x] * batch_duplication, dim=0)
                y = torch.cat([y] * batch_duplication, dim=0)

            x, y = self.forward_train(x, y)
            return x, y
            
        else:
            x = self.forward_eval(x)
            return x, y

    def forward_train(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        lod = Octree3DNoStates(x.permute(0, 4, 1, 2, 3), self.octree_res) #lod in BCHWD
        #lod.plot("hippocampus_octree.pdf")
        #exit()
        x = lod.levels_of_detail[-1].permute(0, 2,3,4, 1)
        x = self.make_seed(x)
        #x: BHWDC
        x = x.permute(0, 4, 1, 2, 3)
        # x: BCHWD
        x = x.to(self.device)
        #x: BCHWD

        if self.patch_sizes[-1] is not None:
            h_start = self.my_rand_int(0, x.shape[2]-self.patch_sizes[-1][0])
            w_start = self.my_rand_int(0, x.shape[3]-self.patch_sizes[-1][1])
            d_start = self.my_rand_int(0, x.shape[4]-self.patch_sizes[-1][2])
            current_patch = np.array([[h_start, w_start, d_start], 
                                      [self.patch_sizes[-1][0] + h_start, 
                                       self.patch_sizes[-1][1] + w_start, 
                                       self.patch_sizes[-1][2] + d_start]
                                     ])
            x = x[:, :, current_patch[0,0]:current_patch[1,0],
                    current_patch[0,1]:current_patch[1,1],
                    current_patch[0,2]:current_patch[1,2]]
        else:
            current_patch = np.array([[0,0,0], [*self.octree_res[-1]]])

        for level in list(range(len(lod.levels_of_detail)))[::-1]: #micro to macro (low res to high res)
            x = x.permute(0, 2,3,4, 1)
            #x: BHWDC

            if self.separate_models:
                x = self.backbone_ncas[level](x, steps=self.inference_steps[level], fire_rate=self.fire_rate)
            else:
                x = self.backbone_nca(x, steps=self.inference_steps[level], fire_rate=self.fire_rate)

            #x: BHWDC
            x = x.permute(0, 4, 1, 2, 3)
            # x: BCHWD

            x = x[:, self.input_channels:]

            if level > 0:
                #upscale x
                x = torch.nn.Upsample(scale_factor=tuple(self.computed_upsampling_scales[level-1][0]), 
                                      mode='nearest')(x)
                current_patch *= self.computed_upsampling_scales[level-1]
            
                #cut out patch from input_channels
                if self.patch_sizes[level-1] is not None:
                    h_start = self.my_rand_int(current_patch[0,0], current_patch[1,0] - self.patch_sizes[level-1][0])
                    w_start = self.my_rand_int(current_patch[0,1], current_patch[1,1] - self.patch_sizes[level-1][1])
                    d_start = self.my_rand_int(current_patch[0,2], current_patch[1,2] - self.patch_sizes[level-1][2])

                    x = x[:, :, 
                          h_start - current_patch[0,0]:h_start - current_patch[0,0] + self.patch_sizes[level-1][0],
                          w_start - current_patch[0,1]:w_start - current_patch[0,1] + self.patch_sizes[level-1][1],
                          d_start - current_patch[0,2]:d_start - current_patch[0,2] + self.patch_sizes[level-1][2]]
                    
                    current_patch = np.array([[h_start, w_start, d_start], 
                                            [self.patch_sizes[level-1][0] + h_start, 
                                            self.patch_sizes[level-1][1] + w_start, 
                                            self.patch_sizes[level-1][2] + d_start]
                                            ])
                #combine with input_channels
                input_channels = lod.levels_of_detail[level-1]
                input_channels = input_channels[:, :, 
                                                current_patch[0,0]:current_patch[1,0],
                                                current_patch[0,1]:current_patch[1,1],
                                                current_patch[0,2]:current_patch[1,2]]
                x = torch.cat([input_channels, x], dim=1)
        
        #outputs: BHWDC

        y = y[:,current_patch[0,0]:current_patch[1,0],
                current_patch[0,1]:current_patch[1,1],
                current_patch[0,2]:current_patch[1,2], :]
        

        x = x.permute(0, 2,3,4, 1)
        x = x[..., :self.output_channels]

        print(current_patch)

        return x, y
    
    @torch.no_grad()
    def forward_eval(self, x: torch.Tensor):
        temp = self.patch_sizes
        self.patch_sizes = [None] * len(self.patch_sizes)
        out, _ = self.forward_train(x, x)
        self.patch_sizes = temp
        return out
    
    def create_inference_series(self, x: torch.Tensor, steps=None):
        assert False, "Not implemented yet"
        #x: BCHWD
        x = x.permute(0, 2,3,4, 1)
        #x: BHWDC
        x = self.make_seed(x)
        x = x.permute(0, 4, 1, 2, 3)
        # x: BCHWD
        x = x.to(self.device)
        lod = Octree3DNoStates(x, self.octree_res)
        
        inference_series = [] #list of BHWDC tensors

        for level in list(range(len(lod.levels_of_detail)))[::-1]:
            x = lod.levels_of_detail[level]
            #x: BCHWD
            x = x.permute(0, 2,3,4, 1)
            #x: BHWDC
            inference_series.append(x)
            
            if self.separate_models:
                x = self.backbone_ncas[level](x, steps=self.inference_steps[level], fire_rate=self.fire_rate)
            else:
                x = self.backbone_nca(x, steps=self.inference_steps[level], fire_rate=self.fire_rate)

            inference_series.append(x)
            #x: BHWDC
            x = x.permute(0, 4, 1, 2, 3)
            # x: BCHWD

            lod.levels_of_detail[level] = x
            if level > 0:
                lod.upscale_states(level)

        outputs = lod.levels_of_detail[0]
        return inference_series
    
    def my_rand_int(self, low, high):
        if high == low:
            return low
        return random.randint(low, high)
        #return np.random.randint(low, high)
    

class Octree3DNoStates:
    @torch.no_grad()
    def __init__(self, init_batch: torch.Tensor, octree_res: list[int]) -> None:

        assert init_batch.ndim == 5, f"init_batch must be BCHWD tensor, got shape {init_batch.shape}"
        
        self.levels_of_detail = [init_batch]
        assert init_batch.shape[2:] == octree_res[0], f"init_batch must have shape {octree_res[0]}, got shape {init_batch.shape[2:]}"

        for resolution in octree_res[1:]:
            lower_res = F.interpolate(self.levels_of_detail[-1], size=resolution)
            self.levels_of_detail.append(lower_res)
                                     

    def plot(self, output_path: str = 'octree.pdf') -> None:
        fig, axs = plt.subplots(1, len(self.levels_of_detail), figsize=(20, 20))
        for i, img in enumerate(self.levels_of_detail):
            depth = img.shape[4]
            axs[i].imshow(img[0, 0, :, :, depth//2].cpu(), cmap='gray')
        plt.savefig(output_path, bbox_inches='tight')