import einops
import torch
import torch.nn as nn
from src.models.Model_BackboneNCA import BackboneNCA
import torchio as tio
import random
import torch.nn.functional as F

import matplotlib.pyplot as plt

class OctreeNCAV2(nn.Module):
    r"""Implementation of the backbone octree NCA
    """
    def __init__(self, channel_n, fire_rate, device, steps=64, hidden_size=128, input_channels=1, output_channels=1, batch_duplication: int = 1,
                 kernel_size=None,
                 octree_res_and_steps: list=None, separate_models: bool=False,
                 compile: bool=False,
                 track_running_stats: bool=False):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
        """
        super(OctreeNCAV2, self).__init__()

        assert track_running_stats == False, "track_running_stats must be False"

        self.channel_n = channel_n
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.device = device
        self.fire_rate = fire_rate
        self.steps = steps
        self.batch_duplication = batch_duplication

        self.octree_res = [r_s[0] for r_s in octree_res_and_steps]
        self.inference_steps = [r_s[1] for r_s in octree_res_and_steps]

        self.separate_models = separate_models
        if separate_models:
            if isinstance(kernel_size, list):
                assert len(kernel_size) == len(octree_res_and_steps), "kernel_size must have same length as octree_res_and_steps"
            else:
                kernel_size = [kernel_size] * len(octree_res_and_steps)
            
            self.backbone_ncas = nn.ModuleList([BackboneNCA(channel_n=channel_n, fire_rate=fire_rate, 
                                                            device=device, hidden_size=hidden_size, 
                                                            input_channels=input_channels, 
                                                            kernel_size=kernel_size[i]) for i in range(len(octree_res_and_steps))])
        else:
            self.backbone_nca = BackboneNCA(channel_n=channel_n, fire_rate=fire_rate, device=device, hidden_size=hidden_size, input_channels=input_channels)

    def make_seed(self, x):
        seed = torch.zeros((x.shape[0], self.channel_n, x.shape[2], x.shape[3]), dtype=torch.float32, device=self.device)
        seed[:, :self.input_channels, :, :] = x
        return seed

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, batch_duplication: int = 1):
        #x: BCHW
        #y: BCHW
        
        assert x.shape[1] == self.input_channels, f"x shape {x.shape} != input_channels {self.input_channels}"
        assert y.shape[1] == self.output_channels, f"y shape {y.shape} != output_channels {self.output_channels}"

        assert x.shape[2:] == y.shape[2:], f"x shape {x.shape} != y shape {y.shape}"

        x = self.make_seed(x).to(self.device)
        x = einops.rearrange(x, 'b c h w -> b h w c')
        y = einops.rearrange(y, 'b c h w -> b h w c')
        y = y.to(self.device)

        #x: BHWC
        #y: BHWC

        if self.training:
            if self.batch_duplication != 1:
                x = torch.cat([x] * self.batch_duplication, dim=0)
                y = torch.cat([y] * self.batch_duplication, dim=0)

            x, y = self.forward_train(x, y)
            assert x.shape == y.shape, f"segmentation shape {x.shape} != y shape {y.shape}"
            return x, y
            
        else:
            x = self.forward_eval(x)
            return x, y

    def forward_train(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device)

        lod = Octree(x, self.input_channels, self.octree_res)

        for level in list(range(len(lod.levels_of_detail)))[::-1]: #micro to macro (low res to high res)
            x = lod.levels_of_detail[level]
            #x: BHWC
            if self.separate_models:
                x = self.backbone_ncas[level](x, steps=self.inference_steps[level], fire_rate=self.fire_rate)
            else:
                x = self.backbone_nca(x, steps=self.inference_steps[level], fire_rate=self.fire_rate)
            lod.levels_of_detail[level] = x
            if level > 0:
                lod.upscale_states(level)
        
        outputs = lod.levels_of_detail[0]


        segmentation = outputs[..., self.input_channels:self.input_channels+self.output_channels]

        return segmentation, y
    
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


        inference_series = []

        for level in list(range(len(lod.levels_of_detail)))[::-1]:
            x = lod.levels_of_detail[level]
            inference_series.append(x)
            x = self.backbone_nca(x, steps=self.inference_steps[level], fire_rate=self.fire_rate)
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
    def __init__(self, init_batch: torch.Tensor, input_channels: int, octree_res: list[tuple]) -> None:
        self.input_channels = input_channels
        self.resolutions = octree_res

        assert init_batch.ndim == 4, "init_batch must be BHWC tensor"


        
        self.levels_of_detail = []
        for res in self.resolutions:
            # create temp with BCHW order
            temp = einops.rearrange(init_batch, 'b h w c -> b c h w')
            lower_res = F.interpolate(temp, size=res)
            lower_res = einops.rearrange(lower_res, 'b c h w -> b h w c')
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
        temp = einops.rearrange(temp, 'b h w c -> b c h w')
        upsampled_states = F.interpolate(temp, size=self.resolutions[from_level-1], mode="nearest")
        upsampled_states = einops.rearrange(upsampled_states, 'b c h w -> b h w c')


        self.levels_of_detail[from_level-1] = torch.cat([self.levels_of_detail[from_level-1][..., :self.input_channels], upsampled_states], dim=-1)