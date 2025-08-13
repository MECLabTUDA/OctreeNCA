from src.models.Model_BasicNCA3D import BasicNCA3D
import torch
import numpy as np
from typing import Callable, Dict
from src.utils.ModelAstAnnotations import forward_vis
from abc import ABC
"""
DEPRECATED DEPRECATED DEPRECATED
DO NOT USE
DO NOT USE
DO NOT USE
"""
class VisualizationModel(ABC):
    def set_state_dict(self, dict: Dict[int, np.ndarray]):
        raise NotImplementedError()

    def export_state_dict(self) -> Dict[int, np.ndarray]:
        raise NotImplementedError()
    
    def set_instrumentation_function(self, function: Callable[[np.ndarray, int], bool]):
        raise NotImplementedError()
    
class BasicNCA3DVis(BasicNCA3D, VisualizationModel):
    """
    Wrapper class for BasicNCA3D. 

    Handles instrumentation of the forward method to allow for visualization
    """
    instrumentation_function: Callable[[np.ndarray, int], bool] = None
    steps_dict: Dict[int, np.ndarray] = None

    def __init__(self, channel_n, fire_rate, device, instrumentation_function: Callable[[np.ndarray, int], bool] = None, hidden_size=128, input_channels=1, init_method="standard", kernel_size=7, groups=False):
        super().__init__(channel_n, fire_rate, device, hidden_size, input_channels, init_method, kernel_size, groups)

    def set_state_dict(self, dict: Dict[int, np.ndarray]):
        self.steps_dict = dict

    def export_state_dict(self) -> Dict[int, np.ndarray]:
        ret = self.steps_dict
        self.steps_dict = None
        return ret
    
    def set_instrumentation_function(self, function: Callable[[np.ndarray, int], bool]):
        self.instrumentation_function = function
    
    def forward(self, x, steps=10, fire_rate=0.5):
        r"""Forward function applies update function step times leaving input channels unchanged
            #Args:
                x: image
                steps: number of steps to run update
                fire_rate: random activation rate of each cell
        """
        
        if not self.steps_dict is None:
            self.steps_dict[0] = x.clone().detach().cpu().numpy().squeeze()
        if not self.instrumentation_function is None:
            self.instrumentation_function(0, x.clone().detach().cpu().numpy().squeeze())
        for step in range(steps):
            #aaaaaaaaaaaaaaaaaaaaaaa
            x2 = self.update(x, fire_rate).clone() #[...,3:][...,3:]
            x = torch.concat((x[...,0:self.input_channels], x2[...,self.input_channels:]), 4)
            if not self.steps_dict is None:
                self.steps_dict[step + 1] = x.clone().detach().cpu().numpy().squeeze()
            if not self.instrumentation_function is None:
                self.instrumentation_function(step + 1, x.clone().detach().cpu().numpy().squeeze())
        #aaaaaaaaaaa
        return x