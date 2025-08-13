from typing import Type, Dict, Any, TYPE_CHECKING, List, Union, TypeVar, Optional, Callable
from torch.utils.data import Dataset
import torch
import torchio
import numpy as np
from src.agents.Agent import BaseAgent
from PIL import Image
import types
from src.datasets.Data_Instance import Data_Container
from src.utils.DatasetClassCreator import Extended_Loadable_Dataset_Annotations
from src.utils.Experiment_vis import Experiment_vis
from src.utils.BasicNCA3DVis import VisualizationModel
from src.utils.ModelAstAnnotations import alter_forward_function
from src.utils.helper import merge_img_label_gt
import torch.nn as nn
import os
from os.path import join
import copy
"""
Same Idea as DatasetCreator and Agent Creator. 
Automatically generate needed classes dynamically at runtime to guarantee compaitibility with existing codebase.
"""

class Extended_Visualization_Model_Annotations(nn.Module):
    if TYPE_CHECKING:
        def __init__(self, channel_n, fire_rate, device, instrumentation_function: Callable[[np.ndarray, int], bool] = None, hidden_size=128, input_channels=1, init_method="standard", kernel_size=7, groups=False):
            
            pass
            
        def set_state_dict(self, dict: Dict[int, np.ndarray]):
            pass

        def export_state_dict(self) -> Dict[int, np.ndarray]:
            pass
        
        def set_instrumentation_function(self, function: Callable[[np.ndarray, int], bool]):
            pass

        def forward(self, x, steps=10, fire_rate=0.5):
            r"""Forward function applies update function step times leaving input channels unchanged
                #Args:
                    x: image
                    steps: number of steps to run update
                    fire_rate: random activation rate of each cell
            """
            pass


class VisualizationModelCreator():
    
    @classmethod
    def create_visualization_model(cls, base_class: Type[nn.Module]) -> Type[Extended_Visualization_Model_Annotations]:
        
        def anonymous_constructor(self, *args, **kwargs): 
            super(NewVisualizationModelClass, self).__init__(*args, **kwargs)
            
            
        def set_state_dict(self, dict: Dict[int, np.ndarray]):
            self.steps_dict = dict

        def export_state_dict(self) -> Dict[int, np.ndarray]:
            ret = self.steps_dict
            self.steps_dict = None
            return ret
    
        def set_instrumentation_function(self, function: Callable[[np.ndarray, int], bool]):
            self.instrumentation_function = function
        
        
        
            
        
        NewVisualizationModelClass = type("ConstructorInitializedDataset", (base_class, ), {
            # constructor
            "__init__": anonymous_constructor, 
            
            # function members
            "set_state_dict": set_state_dict,
            "export_state_dict": export_state_dict,
            "set_instrumentation_function": set_instrumentation_function,
            "forward": alter_forward_function(base_class.forward),
            # object attribes
            "steps_dict": None,
            # Dataset has to be of type Extended loadable Dataset.
            "instrumentation_function": None,
            
            
        })
        return NewVisualizationModelClass







     
