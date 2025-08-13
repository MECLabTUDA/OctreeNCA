from matplotlib import pyplot as plt
import configs.default
import configs.models
import configs.models.prostate_unet
import configs.tasks
import configs.tasks.segmentation
import configs.trainers
import configs.trainers.vitca
from src.datasets.Dataset_DAVIS import Dataset_DAVIS
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.utils.BaselineConfigs import EXP_OctreeNCA3D, EXP_OctreeNCA3D_superres, EXP_UNet2D, EXP_UNet3D
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.datasets.png_seg_Dataset import png_seg_Dataset
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
import octree_vis, os, torch, shutil
import pickle as pkl
from src.datasets.Dataset_CholecSeg import Dataset_CholecSeg
from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed

import torchio as tio

import configs


print(ProjectConfiguration.STUDY_PATH)

study_config = {
    'experiment.name': r'prostate_unet',
    'experiment.description': "UNetSegmentation",

    'model.output_channels': 1,
}
study_config = study_config | configs.models.prostate_unet.prostate_unet_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.prostate.prostate_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config

study_config['trainer.ema'] = False
study_config['performance.compile'] = False

study_config['trainer.batch_size'] = 3

study_config['experiment.dataset.input_size'] = [320, 320, 24]
study_config['experiment.dataset.patchify'] = True
study_config['experiment.dataset.patchify.foreground_oversampling_probability'] = 0.5
study_config['experiment.dataset.patchify.patch_size'] = [160, 160, 24]

study_config['model.eval.patch_wise'] = True

study_config['model.num_encoding_blocks'] = 2
study_config['model.out_channels_first_layer'] = 4

study = Study(study_config)


exp = EXP_UNet3D().createExperiment(study_config, detail_config={}, dataset_class=Dataset_NiiGz_3D, dataset_args={})
study.add_experiment(exp)

study.run_experiments()
study.eval_experiments()