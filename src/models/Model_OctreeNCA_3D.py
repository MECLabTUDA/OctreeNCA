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

from src.models.Model_BasicNCA3D_fast import BasicNCA3DFast
from src.models.Model_VitCA3D import ViTCA3D

class OctreeNCA3D(nn.Module):
    r"""Implementation of M3D-NCA
    """
    def __init__(self, config: dict):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
        """
        super(OctreeNCA3D, self).__init__()
        assert config['model.batchnorm_track_running_stats'] is False, "batchnorm_track_running_stats must be False"

        self.channel_n = config['model.channel_n']
        self.input_channels = config['model.input_channels']
        self.output_channels = config['model.output_channels']
        self.device = config['experiment.device']
        self.fire_rate = config['model.fire_rate']

        
        octree_res_and_steps = config['model.octree.res_and_steps']
        separate_models = config['model.octree.separate_models']
        kernel_size = config['model.kernel_size']

        normalization = config.get("model.normalization", "batch")

        self.octree_res = [tuple(r_s[0]) for r_s in octree_res_and_steps]
        self.inference_steps = [r_s[1] for r_s in octree_res_and_steps]

        self.separate_models = separate_models

        if config['performance.compile']:
            torch.set_float32_matmul_precision('high')
        
        if separate_models:
            if isinstance(kernel_size, list):
                assert len(kernel_size) == len(octree_res_and_steps), "kernel_size must have same length as octree_res_and_steps"
            else:
                kernel_size = [kernel_size] * len(octree_res_and_steps)
        else:
            assert isinstance(kernel_size, int), "kernel_size must be an integer"


        backbone_class = eval(config.get("model.backbone_class", "BasicNCA3D"))
        assert backbone_class in [BasicNCA3D, BasicNCA3DFast], f"backbone_class must be either BasicNCA3D, got {backbone_class}"

        if separate_models:
            if config["model.vitca"]:
                self.backbone_ncas = []
                for l in range(len(octree_res_and_steps)):
                    conv_size = kernel_size[l]
                    self.backbone_ncas.append(ViTCA3D(patch_size=1, depth=config["model.vitca.depth"], heads=config["model.vitca.heads"],
                                           mlp_dim=config["model.vitca.mlp_dim"], dropout=config["model.vitca.dropout"], 
                                           cell_in_chns=self.input_channels, cell_out_chns=self.output_channels, 
                                           cell_hidden_chns=self.channel_n - self.input_channels - self.output_channels, 
                                           embed_cells=config["model.vitca.embed_cells"], embed_dim=config["model.vitca.embed_dim"],
                                           embed_dropout=config["model.vitca.embed_dropout"], 
                                           localize_attn=True, localized_attn_neighbourhood=[conv_size, conv_size, conv_size], 
                                           device=config["experiment.device"]
                                           ))
                self.backbone_ncas = nn.ModuleList(self.backbone_ncas)
            else:
                self.backbone_ncas = nn.ModuleList([backbone_class(channel_n=self.channel_n, fire_rate=self.fire_rate, device=self.device, 
                                                            hidden_size=config['model.hidden_size'], input_channels=self.input_channels, kernel_size=kernel_size[l],
                                                            inplace_relu=config['performance.inplace_operations'], normalization=normalization) 
                                                            for l in range(len(octree_res_and_steps))])
            if config['performance.compile']:
                for i, model in enumerate(self.backbone_ncas):
                    self.backbone_ncas[i] = torch.compile(model)
        else:
            if config["model.vitca"]:
                conv_size = config["kernel_size"]
                self.backbone_nca = ViTCA3D(patch_size=1, depth=config["model.vitca.depth"], heads=config["model.vitca.heads"],
                                           mlp_dim=config["model.vitca.mlp_dim"], dropout=config["model.vitca.dropout"], 
                                           cell_in_chns=self.input_channels, cell_out_chns=self.output_channels, 
                                           cell_hidden_chns=self.channel_n - self.input_channels - self.output_channels,
                                           embed_cells=config["model.vitca.embed_cells"], embed_dim=config["model.vitca.embed_dim"],
                                           embed_dropout=config["model.vitca.embed_dropout"], 
                                           localize_attn=True, localized_attn_neighbourhood=[conv_size, conv_size, conv_size], device=config["device"]
                                           )
            else:
                self.backbone_nca = backbone_class(channel_n=self.channel_n, fire_rate=self.fire_rate, device=self.device, hidden_size=config['model.hidden_size'], 
                                            input_channels=self.input_channels, kernel_size=kernel_size, inplace_relu=config['performance.inplace_operations'], 
                                            normalization=normalization)
            if config['performance.compile']:
                self.backbone_nca = torch.compile(self.backbone_nca)

    def make_seed(self, x):
        # x: BHWDC
        seed = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3], self.channel_n), dtype=torch.float32, device=self.device)
        seed[..., 0:x.shape[-1]] = x 
        # seed: BHWDC 
        return seed


    def get_gpu_memory(self):
        output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
        try:
            memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
        except sp.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
        # print(memory_use_values)
        return memory_use_values

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, batch_duplication=1):
        #x: BHWDC
        #y: BHWDC
        x = self.make_seed(x).to(self.device)
        #x: BHWDC
        x = x.permute(0, 4, 1, 2, 3)
        # x: BCHWD
        x = x.to(self.device)

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
        lod = Octree3D(x, self.input_channels, self.octree_res)
        #lod.plot("hippocampus_octree.pdf")
        #exit()

        #print("LOD")
        #[print(x.shape) for x in lod.levels_of_detail]
        #print("LOD")

        for level in list(range(len(lod.levels_of_detail)))[::-1]: #micro to macro (low res to high res)
            x = lod.levels_of_detail[level]
            #x: BCHWD

            x = x.permute(0, 2,3,4, 1)
            #x: BHWDC

            if self.separate_models:
                x = self.backbone_ncas[level](x, steps=self.inference_steps[level], fire_rate=self.fire_rate)
            else:
                x = self.backbone_nca(x, steps=self.inference_steps[level], fire_rate=self.fire_rate)

            #x: BHWDC
            x = x.permute(0, 4, 1, 2, 3)
            # x: BCHWD

            lod.levels_of_detail[level] = x


            if level > 0:
                lod.upscale_states(level)
        
        outputs = lod.levels_of_detail[0].permute(0, 2,3,4, 1)
        #outputs: BHWDC

        return outputs[..., self.input_channels:self.input_channels+self.output_channels], y
    
    @torch.no_grad()
    def forward_eval(self, x: torch.Tensor):
        out, _ = self.forward_train(x, x)
        return out
    
    def create_inference_series(self, x: torch.Tensor, steps=None):
        #x: BCHWD
        x = x.permute(0, 2,3,4, 1)
        #x: BHWDC
        x = self.make_seed(x)
        x = x.permute(0, 4, 1, 2, 3)
        # x: BCHWD
        x = x.to(self.device)
        lod = Octree3D(x, self.input_channels, self.octree_res)
        
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
    

class Octree3D:
    @torch.no_grad()
    def __init__(self, init_batch: torch.Tensor, input_channels: int, octree_res: list[int]) -> None:
        self.input_channels = input_channels
        self.octree_res = octree_res

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

    def upscale_states(self, from_level: int) -> None:
        assert from_level in range(1, len(self.levels_of_detail)), "from_level must be in range(1, len(levels_of_detail))"
        temp = self.levels_of_detail[from_level]
        #temp: BCHWD
        temp = temp[:, self.input_channels:]

        upsampled_states = torch.nn.Upsample(size=self.octree_res[from_level-1], mode='nearest')(temp)

        self.levels_of_detail[from_level-1] = torch.cat([self.levels_of_detail[from_level-1][:, :self.input_channels], upsampled_states], dim=1)