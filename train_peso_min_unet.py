from matplotlib import pyplot as plt
import configs.default
import configs.models
import configs.tasks
import configs.tasks.segmentation
import configs.trainers
import configs.trainers.vitca
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.datasets.Dataset_DAVIS import Dataset_DAVIS
from src.datasets.Dataset_PESO import Dataset_PESO
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.utils.BaselineConfigs import EXP_OctreeNCA, EXP_OctreeNCA3D, EXP_OctreeNCA3D_superres, EXP_min_UNet2D
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
from src.datasets.png_seg_Dataset import png_seg_Dataset
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
import octree_vis, os, torch, shutil
import pickle as pkl
from src.datasets.Dataset_CholecSeg import Dataset_CholecSeg
from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed

import torchio as tio

import configs

print(pc.STUDY_PATH)

study_config = {
    'experiment.name': r'peso',
    'experiment.description': "MinUNet2DSegmentation",

    'model.output_channels': 1,
}
study_config = study_config | configs.models.min_unet.min_unet_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.peso.peso_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config

study_config['experiment.logging.also_eval_on_train'] = False
study_config['experiment.logging.evaluate_interval'] = study_config['trainer.n_epochs']+1
study_config['experiment.task.score'] = ["src.scores.PatchwiseDiceScore.PatchwiseDiceScore",
                                         "src.scores.PatchwiseIoUScore.PatchwiseIoUScore"]


study_config['model.arch'] = "UNet"
study_config['model.encoder_name'] = "efficientnet-b0"
study_config['model.encoder_depth'] = 3
#study_config['model.decoder_channels'] = [256, 128, 64]
study_config['model.decoder_channels'] = [128, 64, 32]

study_config['experiment.name'] += f"_{study_config['model.arch']}_{study_config['model.encoder_name']}_D{study_config['model.encoder_depth']}_0"


study_config['trainer.ema'] = False
study_config['trainer.batch_size'] = 10

#study_config['experiment.dataset.img_level'] = 0

#study_config['trainer.num_steps_per_epoch'] = 20


study = Study(study_config)


exp = EXP_min_UNet2D().createExperiment(study_config, detail_config={}, dataset_class=Dataset_PESO, dataset_args={
                                                            'patches_path': os.path.join(pc.FILER_BASE_PATH, study_config['experiment.dataset.patches_path']),
                                                            'patch_size': study_config['experiment.dataset.input_size'],
                                                            'path': os.path.join(pc.FILER_BASE_PATH, study_config['experiment.dataset.img_path']),
                                                            'img_level': study_config['experiment.dataset.img_level']
                                                      })
study.add_experiment(exp)

study.run_experiments()


study.eval_experiments(export_prediction=True)
#figure = octree_vis.visualize(study.experiments[0])
#assert(isinstance(exp.agent, MedNCAAgent))
#exp.agent.getAverageDiceScore(pseudo_ensemble=True, ood_augmentation=None, output_name=None, export_prediction=True)

