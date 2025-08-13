import torch
import torch.nn as nn
from src.models.Model_BackboneNCA import BackboneNCA
import torchio as tio
import random
import torch.nn.functional as F

import matplotlib.pyplot as plt

class OctreeNCA(nn.Module):
    r"""Implementation of the backbone octree NCA
    """
    def __init__(self, channel_n, fire_rate, device, steps=64, hidden_size=128, input_channels=1, output_channels=1, batch_duplication: int = 1):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
        """
        super(OctreeNCA, self).__init__()

        self.channel_n = channel_n
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.device = device
        self.fire_rate = fire_rate
        self.steps = steps
        self.batch_duplication = batch_duplication

        self.backbone_nca = BackboneNCA(channel_n=channel_n, fire_rate=fire_rate, device=device, hidden_size=hidden_size, input_channels=input_channels)

    def make_seed(self, x):
        seed = torch.zeros((x.shape[0], self.channel_n, x.shape[2], x.shape[3]), dtype=torch.float32, device=self.device)
        seed[:, :x.shape[self.input_channels], :, :] = x
        return seed

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        #x: BCHW
        #y: BCHW
        x = self.make_seed(x).to(self.device)
        x = x.transpose(1,3)
        y = y.transpose(1,3)
        y = y.to(self.device)

        #x: BHWC
        #y: BHWC

        if self.training:
            if self.batch_duplication != 1:
                x = torch.cat([x] * self.batch_duplication, dim=0)
                y = torch.cat([y] * self.batch_duplication, dim=0)

            x, y = self.forward_train(x, y)
            return x, y
            
        else:
            x = self.forward_eval(x)
            return x, y

    def forward_train(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device)

        lod = Octree(x, self.input_channels)

        steps_arr = [2] * len(lod.levels_of_detail)
        steps_arr[-1] = 16

        for level in list(range(len(lod.levels_of_detail)))[::-1]: #micro to macro (low res to high res)
            x = lod.levels_of_detail[level]
            #x: BHWC
            x = self.backbone_nca(x, steps=steps_arr[level], fire_rate=self.fire_rate)
            lod.levels_of_detail[level] = x
            if level > 0:
                lod.upscale_states(level)
        
        outputs = lod.levels_of_detail[0]


        return outputs[..., self.input_channels:self.input_channels+self.output_channels], y
    
    @torch.no_grad()
    def forward_eval(self, x: torch.Tensor):
        out, _ = self.forward_train(x, x)
        return out
    
    def create_inference_series(self, x: torch.Tensor, steps: int = 64):
        # get array of BHWC
        x = self.make_seed(x).to(self.device)
        x = x.transpose(1,3)
        lod = Octree(x, self.input_channels)

        lod.plot()

        steps_arr = [2] * len(lod.levels_of_detail)
        steps_arr[-1] = 16

        inference_series = []

        for level in list(range(len(lod.levels_of_detail)))[::-1]:
            x = lod.levels_of_detail[level]
            inference_series.append(x)
            x = self.backbone_nca(x, steps=steps_arr[level], fire_rate=self.fire_rate)
            inference_series.append(x)
            lod.levels_of_detail[level] = x
            if level > 0:
                lod.upscale_states(level)

        outputs = lod.levels_of_detail[0]
        return inference_series
        

    def resize4d(self, img: torch.Tensor, size: tuple = (64,64), factor: int = 4, label: bool = False) -> torch.Tensor:
        r"""Resize input image
            #Args
                img: 4d image to rescale
                size: image size
                factor: scaling factor
                label: is Label?
        """
        if label:
            transform = tio.Resize((size[0], size[1], -1), image_interpolation='NEAREST')
        else:
            transform = tio.Resize((size[0], size[1], -1))
        img = transform(img)
        return img
    


class Octree:
    @torch.no_grad()
    def __init__(self, init_batch: torch.Tensor, input_channels: int) -> None:
        self.input_channels = input_channels

        assert init_batch.ndim == 4, "init_batch must be BHWC tensor"
        assert init_batch.shape[1] == init_batch.shape[2], "init_batch must be square"
        
        self.levels_of_detail = [init_batch]
        while self.levels_of_detail[-1].shape[1] > 16:
            # create temp with BCHW order
            temp = self.levels_of_detail[-1].permute(0, 3, 1, 2)
            lower_res = F.avg_pool2d(temp, 2)
            lower_res = lower_res.permute(0, 2, 3, 1)
            self.levels_of_detail.append(lower_res)

    def plot(self, output_path: str = 'octree.pdf') -> None:
        fig, axs = plt.subplots(1, len(self.levels_of_detail), figsize=(20, 20))
        for i, img in enumerate(self.levels_of_detail):
            axs[i].imshow(img[0, :, :, 0].cpu(), cmap='gray')
        plt.savefig(output_path, bbox_inches='tight')

    def upscale_states(self, from_level: int) -> None:
        assert from_level in range(1, len(self.levels_of_detail)), "from_level must be in range(1, len(levels_of_detail))"
        temp = self.levels_of_detail[from_level]
        temp = temp[..., self.input_channels:]
        temp = temp.permute(0, 3, 1, 2) # BHWC -> BCHW
        upsampled_states = torch.nn.Upsample(scale_factor=2, mode='nearest')(temp)
        upsampled_states = upsampled_states.permute(0, 2, 3, 1) # BCHW -> BHWC


        self.levels_of_detail[from_level-1] = torch.cat([self.levels_of_detail[from_level-1][..., :self.input_channels], upsampled_states], dim=-1)