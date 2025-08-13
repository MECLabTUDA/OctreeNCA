from matplotlib import pyplot as plt
from src.datasets.Dataset_DAVIS import Dataset_DAVIS
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.utils.BaselineConfigs import EXP_OctreeNCA3D, EXP_OctreeNCA3D_superres
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
    'experiment.name': r'prostate_m3d_fast',
    'experiment.description': "M3dSegmentation",

    'model.output_channels': 1,
}
study_config = study_config | configs.models.prostate_m3d.prostate_m3d_nca_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.prostate.prostate_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config

study_config['trainer.ema'] = False
study_config['performance.compile'] = False

study_config['trainer.batch_size'] = 3
study_config['trainer.batch_duplication'] = 2

study_config['model.backbone_class'] = "BasicNCA3DFast"

study = Study(study_config)

ood_augmentation = None
output_name = None
severity = 6
#ood_augmentation = tio.RandomGhosting(num_ghosts=severity, intensity=0.5 * severity)
if ood_augmentation != None:
    output_name = f"{ood_augmentation.__class__.__name__}_{severity}.csv"

exp = EXP_OctreeNCA3D().createExperiment(study_config, detail_config={}, dataset_class=Dataset_NiiGz_3D, dataset_args={})
study.add_experiment(exp)

study.run_experiments()
study.eval_experiments(ood_augmentation=ood_augmentation, output_name=output_name)
#figure = octree_vis.visualize(study.experiments[0])


