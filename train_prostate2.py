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
    'experiment.name': r'prostateAbl_none_5_12',
    'experiment.description': "OctreeNCASegmentation",

    'model.output_channels': 1,
}
study_config = study_config | configs.models.prostate.prostate_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.prostate.prostate_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config

#study_config['experiment.logging.evaluate_interval'] = 1

study_config['trainer.ema'] = True
study_config['performance.compile'] = False
study_config['model.train.loss_weighted_patching'] = False

#study_config['trainer.find_best_model_on'] = "train"
#study_config['trainer.always_eval_in_last_epochs'] = 300

#study_config['model.channel_n'] = 24
#study_config['model.hidden_size'] = 100
#study_config['model.kernel_size'] = [3, 3, 3, 3, 7]
#study_config['model.octree.res_and_steps'] = [[[320,320,24], 20], [[160,160,12], 20], [[80,80,6], 20], [[40,40,6], 20], [[20,20,6], 40]]
#study_config['trainer.optimizer.weight_decay'] = 0.001

#study_config['trainer.losses'] = ["src.losses.IntermediateSupervision.DiceBCELossInterSuperv"]


#study_config['model.train.patch_sizes'] = [[160, 160, 12], None, None, None, None]

study_config['model.backbone_class'] = "BasicNCA3DFast"



study_config['model.normalization'] = "none"    #"none"

steps = 10                                      # 10
alpha = 1.0                                     # 1.0
study_config['model.octree.res_and_steps'] = [[[320,320,24], steps], [[160,160,12], steps], [[80,80,6], steps], [[40,40,6], steps], [[20,20,6], int(alpha * 20)]]


study_config['model.channel_n'] = 16            # 16
study_config['model.hidden_size'] = 64          # 64

study_config['trainer.batch_size'] = 3          # 3

dice_loss_weight = 1.0                          # 1.0


ema_decay = 0.99                                # 0.99
study_config['trainer.ema'] = ema_decay > 0.0
study_config['trainer.ema.decay'] = ema_decay


study_config['trainer.losses'] = ["src.losses.DiceLoss.DiceLoss", "src.losses.BCELoss.BCELoss"]
study_config['trainer.losses.parameters'] = [{}, {}]
study_config['trainer.loss_weights'] = [dice_loss_weight, 2.0-dice_loss_weight]
#study_config['trainer.loss_weights'] = [1.5, 0.5]

study_config['experiment.name'] = f"prostatefAbl_{study_config['model.normalization']}_{steps}_{alpha}_{study_config['model.channel_n']}_{study_config['trainer.batch_size']}_{dice_loss_weight}_{ema_decay}"



study = Study(study_config)

exp = EXP_OctreeNCA3D().createExperiment(study_config, detail_config={}, dataset_class=Dataset_NiiGz_3D, dataset_args={})
study.add_experiment(exp)

study.run_experiments()
study.eval_experiments()
#figure = octree_vis.visualize(study.experiments[0])


study.eval_experiments_ood()
