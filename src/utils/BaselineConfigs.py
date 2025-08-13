import torch
from src.agents.Agent_M3DNCA_GradAccum import M3DNCAAgentGradientAccum
from src.agents.Agent_M3DNCA_superres import M3DNCAAgent_superres
from src.agents.Agent_MedNCA_extrapolation import MedNCAAgent_extrapolation
from src.agents.Agent_MedSeg3D_slicewise import Agent_MedSeg3D_slicewise
from src.losses.WeightedLosses import WeightedLosses
from src.models.SamWrapper2D import SamWrapper2D
from src.models.SamWrapper3D import SamWrapper3D
from src.models.SegFormerWrapper2D import SegFormerWrapper2D
from src.models.UNetWrapper2D import UNetWrapper2D
from src.models.UNetWrapper3D import UNetWrapper3D
from src.models.Model_OctreeNCA_3d_patching import OctreeNCA3DPatch
from src.models.Model_OctreeNCA_3d_patching2 import OctreeNCA3DPatch2
from src.utils.ExperimentWrapper import ExperimentWrapper
from src.utils.Experiment import merge_config
import numpy as np
from ..losses.LossFunctions import DiceBCELoss
from torch.utils.data import Dataset

from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from unet import UNet2D
from src.agents.Agent_UNet import UNetAgent
    
from src.models.Model_M3DNCA import M3DNCA
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
from src.losses.LossFunctions import DiceFocalLoss

class EXP_M3DNCA(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset : Dataset = None):
        config = {
            'description': 'M3DNCA',
            'lr': 16e-4,
            'batch_duplication': 1,
            # Model
            'channel_n': 16,        # Number of CA state channels
            'inference_steps': 20,
            'cell_fire_rate': 0.5,
            'batch_size': 4,
            'hidden_size': 64,
            'train_model':3,
            # Data
            'scale_factor': 4,
            'kernel_size': 7,
            'levels': 2,
        }

        config = merge_config(merge_config(study_config, config), detail_config)
        if dataset is None:
            dataset = Dataset_NiiGz_3D()
        model = M3DNCA(config['channel_n'], config['cell_fire_rate'], device=config['device'], hidden_size=config['hidden_size'], kernel_size=config['kernel_size'], input_channels=config['input_channels'], levels=config['levels'], scale_factor=config['scale_factor'], steps=config['inference_steps'])
        agent = M3DNCAAgent(model)
        loss_function = DiceBCELoss() 

        return super().createExperiment(config, model, agent, dataset, loss_function)
    
from src.models.Model_MedNCA import MedNCA
from src.agents.Agent_MedNCA_Simple  import MedNCAAgent

class EXP_MEDNCA(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset : Dataset = None):
        config = {
            'description': 'MEDNCA',
            'lr': 16e-4,
            'batch_duplication': 1,
            # Model
            'channel_n': 32,        # Number of CA state channels
            'inference_steps': 64,
            'cell_fire_rate': 0.5,
            'batch_size': 12,
            'hidden_size': 128,
            'train_model':1,
            'betas': (0.9, 0.99),
            # Data
            'scale_factor': 4,
            'kernel_size': 3,
            'levels': 2,
            'input_size': (320,320) ,
        }

        config = merge_config(merge_config(study_config, config), detail_config)
        print("CONFIG", config)
        if dataset is None:
            dataset = Dataset_NiiGz_3D(slice=2)
        model = MedNCA(config['channel_n'], config['cell_fire_rate'], device=config['device'], hidden_size=config['hidden_size'], input_channels=config['input_channels'], steps=config['inference_steps'])
        agent = MedNCAAgent(model)
        loss_function = DiceBCELoss() 

        return super().createExperiment(config, model, agent, dataset, loss_function)
  
from src.models.Model_BasicNCA import BasicNCA

class EXP_BasicNCA(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset : Dataset = None):
        config = {
            'description': 'MEDNCA',
            'lr': 16e-4,
            'batch_duplication': 1,
            # Model
            'channel_n': 32,        # Number of CA state channels
            'inference_steps': 64,
            'cell_fire_rate': 0.5,
            'batch_size': 12,
            'hidden_size': 128,
            'train_model':1,
            'betas': (0.9, 0.99),
            # Data
            'scale_factor': 4,
            'kernel_size': 3,
            'levels': 2,
            'input_size': (320,320) ,
        }

        config = merge_config(merge_config(study_config, config), detail_config)
        print("CONFIG", config)
        if dataset is None:
            dataset = Dataset_NiiGz_3D(slice=2)
        model = BasicNCA(config['channel_n'], config['cell_fire_rate'], device=config['device'], hidden_size=config['hidden_size'], input_channels=config['input_channels'], steps=config['inference_steps'])
        agent = MedNCAAgent(model)
        loss_function = DiceBCELoss() 

        return super().createExperiment(config, model, agent, dataset, loss_function)
    

from src.models.Model_OctreeNCA_2d_patching2 import OctreeNCA2DPatch2
from src.models.Model_OctreeNCA import OctreeNCA
from src.models.Model_OctreeNCAV2 import OctreeNCAV2
class EXP_OctreeNCA(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        config = study_config
        if dataset_class is None:
            assert False, "Dataset is None"
        model = OctreeNCA2DPatch2(config)
        
        assert config['model.batchnorm_track_running_stats'] == False
        assert config['trainer.gradient_accumulation'] == False
        assert config['trainer.train_quality_control'] == False

        assert config['experiment.task'] == 'segmentation'
        agent = MedNCAAgent(model)
        loss_function = WeightedLosses(config) 

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)
    

class EXP_OctreeNCA2D_extrapolation(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):

        config = study_config
        if dataset_class is None:
            assert False, "Dataset is None"
        model = OctreeNCA2DPatch2(config)
        
        assert config['model.batchnorm_track_running_stats'] == False
        assert config['trainer.gradient_accumulation'] == False
        assert config['trainer.train_quality_control'] == False

        assert config['experiment.task'] == 'extrapolation'
        agent = MedNCAAgent_extrapolation(model)
        loss_function = WeightedLosses(config)

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)
    
from src.models.Model_OctreeNCA_3D import OctreeNCA3D
class EXP_OctreeNCA3D(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        config = study_config
        
        if dataset_class is None:
            assert False, "Dataset is None"

        if 'model.train.patch_sizes' in config:
            model = OctreeNCA3DPatch2(config)
        else:
            assert False, "deprecated"
            model = OctreeNCA3D(config['channel_n'], config['cell_fire_rate'], device=config['device'], hidden_size=config['hidden_size'], input_channels=config['input_channels'], 
                                output_channels=config['output_channels'], steps=config['inference_steps'],
                                octree_res_and_steps=config['octree_res_and_steps'], separate_models=config['separate_models'],
                                compile=config['compile'], kernel_size=config['kernel_size'])
            
        #print(model)
        #input("model")

        if 'gradient_accumulation' in config and config['gradient_accumulation']:
            agent = M3DNCAAgentGradientAccum(model)
        else:
            agent = M3DNCAAgent(model)
            #agent = Agent_M3D_NCA(model)
        loss_function = WeightedLosses(config)

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)

class EXP_OctreeNCA3D_superres(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        config = study_config

        if dataset_class is None:
            assert False, "Dataset is None"


        model = OctreeNCA3DPatch2(config)
            
        if 'gradient_accumulation' in config and config['gradient_accumulation']:
            assert False, "not implemented"
            agent = M3DNCAAgentGradientAccum(model)
        else:
            agent = M3DNCAAgent_superres(model)
        loss_function = WeightedLosses(config)

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)




from unet import UNet3D
class EXP_UNet3D(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):

        config = study_config
        
        if dataset_class is None:
            assert False, "Dataset is None"

        model_params = {k.replace("model.", ""): v for k, v in config.items() if k.startswith('model.')}
        model_params.pop("output_channels")
        model_params.pop("input_channels")
        model_params.pop("eval.patch_wise") if "eval.patch_wise" in model_params else None
        model = UNet3D(in_channels=config['model.input_channels'], out_classes=config['model.output_channels'], padding=1, **model_params)
        model = UNetWrapper3D(model)
        if config['performance.compile']:
            model.compile()
            
        assert not config.get('gradient_accumulation', False)
        agent = M3DNCAAgent(model)
        loss_function = WeightedLosses(config)

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)

class EXP_SAM3D(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        from sam2.build_sam import build_sam2
        #from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        from sam2.build_sam import build_sam2_video_predictor
        config = study_config
        
        if dataset_class is None:
            assert False, "Dataset is None"
        
        assert not config.get('gradient_accumulation', False)


        assert config['model.input_channels'] == 3
        
        checkpoint = config['model.checkpoint']
        model_cfg = config['model.model_cfg']
        predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda")
        model = SamWrapper3D(predictor)
        
        print("numparams", sum(p.numel() for p in model.parameters()))
        input("Press Enter to continue...")

        if config['performance.compile']:
            model.compile()
        agent = M3DNCAAgent(model)
        loss_function = WeightedLosses(config)

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)
    

class EXP_UNet2D(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        config = study_config
        
        if dataset_class is None:
            assert False, "Dataset is None"
        
        assert not config.get('gradient_accumulation', False)

        model_params = {k.replace("model.", ""): v for k, v in config.items() if k.startswith('model.')}
        model_params.pop("output_channels")
        model_params.pop("input_channels")
        model = UNet2D(in_channels=config['model.input_channels'], out_classes=config['model.output_channels'], padding=1, **model_params)
        model = UNetWrapper2D(model)
        

        if config['performance.compile']:
            model.compile()
        agent = MedNCAAgent(model)
        loss_function = WeightedLosses(config)

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)
    
class EXP_SegFormer2D(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        config = study_config
        
        if dataset_class is None:
            assert False, "Dataset is None"
        
        assert not config.get('gradient_accumulation', False)

        model_params = {k.replace("model.", ""): v for k, v in config.items() if k.startswith('model.')}
        model_params.pop("output_channels")
        model_params.pop("input_channels")
        configuration = SegformerConfig(num_channels=config['model.input_channels'], **model_params)
        model = SegformerForSemanticSegmentation(configuration).to(config['experiment.device'])
        model = SegFormerWrapper2D(model)
        

        if config['performance.compile']:
            model.compile()
        agent = MedNCAAgent(model)
        loss_function = WeightedLosses(config)

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)
    
class EXP_SAM2D(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        config = study_config
        
        if dataset_class is None:
            assert False, "Dataset is None"
        
        assert not config.get('gradient_accumulation', False)

        model_params = {k.replace("model.", ""): v for k, v in config.items() if k.startswith('model.')}
        model_params.pop("output_channels")
        model_params.pop("input_channels")
        assert config['model.input_channels'] == 3
        from segment_anything import SamPredictor, sam_model_registry
        sam = sam_model_registry["vit_h"](checkpoint="<path>/sam_checkpoints/sam_vit_h_4b8939.pth")
        model = SamPredictor(sam.cuda())
        model = SamWrapper2D(model)
        
        print("numparams", sum(p.numel() for p in model.parameters()))
        input("Press Enter to continue...")

        if config['performance.compile']:
            model.compile()
        agent = MedNCAAgent(model)
        loss_function = WeightedLosses(config)

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)

    

class EXP_min_UNet2D(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        config = study_config
        
        if dataset_class is None:
            assert False, "Dataset is None"
        
        assert not config.get('gradient_accumulation', False)
        
        model_params = {k.replace("model.", ""): v for k, v in config.items() if k.startswith('model.')}
        model_params.pop("output_channels")
        model_params.pop("input_channels")
        #model_params.pop("eval.patch_wise")
        model = smp.create_model(in_channels=config['model.input_channels'], classes=config['model.output_channels'],**model_params)
        model = UNetWrapper2D(model)
        
        if config['performance.compile']:
            model.compile()
        agent = MedNCAAgent(model)
        loss_function = WeightedLosses(config)

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)

import segmentation_models_pytorch_3d as smp3d
class EXP_min_UNet3D(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        config = study_config
        
        if dataset_class is None:
            assert False, "Dataset is None"
        
        assert not config.get('gradient_accumulation', False)
        
        model_params = {k.replace("model.", ""): v for k, v in config.items() if k.startswith('model.')}
        model_params.pop("output_channels")
        model_params.pop("input_channels")
        #model_params.pop("eval.patch_wise")
        model = smp3d.create_model(in_channels=config['model.input_channels'], classes=config['model.output_channels'],**model_params, encoder_weights=None)
        model = UNetWrapper3D(model)
        #print(model)
        #input("model")


        if config['performance.compile']:
            model.compile()
        agent = M3DNCAAgent(model)
        loss_function = WeightedLosses(config)

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)
    

import segmentation_models_pytorch as smp
class EXP_min_UNet(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        config = study_config
        
        if dataset_class is None:
            assert False, "Dataset is None"
        
        assert not config.get('gradient_accumulation', False)

        model_params = {k.replace("model.", ""): v for k, v in config.items() if k.startswith('model.')}
        model_params.pop("output_channels")
        model_params.pop("input_channels")
        #model_params.pop("eval.patch_wise")
        model = smp.create_model(in_channels=config['model.input_channels'], classes=config['model.output_channels'],**model_params)

        model = UNetWrapper2D(model)

        if config['performance.compile']:
            model.compile()
        agent = Agent_MedSeg3D_slicewise(model)
        #agent = MedNCAAgent(model)
        loss_function = WeightedLosses(config)

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)

    

