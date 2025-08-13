import torch
import torch.nn as nn
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.models.Model_BasicNCA3D_alive import BasicNCA3D_alive
import torchio as tio
import random
import math
import torch.nn.functional as F
import subprocess as sp
from src.models.Model_M3DNCA import M3DNCA

class M3DNCA_alive(M3DNCA):
    r"""Implementation of M3D-NCA
    """
    def __init__(self, channel_n, fire_rate, device, steps=64, hidden_size=128, input_channels=1, output_channels=1, scale_factor=4, levels=2, kernel_size=7):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
        """
        super(M3DNCA, self).__init__()

        self.channel_n = channel_n
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.device = device
        self.fire_rate = fire_rate
        self.steps = steps
        self.scale_factor = scale_factor
        self.levels = levels
        self.fast_inf = True
        self.margin = 7

        self.model = nn.ModuleList()
        for i in range(self.levels):    
            if i == 0:
                self.model.append(BasicNCA3D(channel_n=channel_n, fire_rate=fire_rate, device=device, hidden_size=hidden_size, input_channels=input_channels, kernel_size=kernel_size))
            else:
                self.model.append(BasicNCA3D_alive(channel_n=channel_n, fire_rate=fire_rate, device=device, hidden_size=hidden_size, input_channels=input_channels, kernel_size=3))

    def make_seed(self, x):
        #seed = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3], self.channel_n), dtype=torch.float32, device=self.device)
        seed = torch.full((x.shape[0], x.shape[1], x.shape[2], x.shape[3], self.channel_n), -6, dtype=torch.float32, device=self.device)
        seed[..., self.input_channels+self.output_channels:] = 0
        seed[..., 0:x.shape[-1]] = x 
        return seed

    #def forward_train(self, x: torch.Tensor, y: torch.Tensor):
    #    outputs, targets = super().forward_train(x, y)
    #    outputs = outputs 
    #    return outputs, targets
    
    #def forward_eval(self, x: torch.Tensor):
    #    outputs = super().forward_eval(x)
    #    outputs = outputs 
    #    return outputs