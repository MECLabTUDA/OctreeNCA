from matplotlib import pyplot as plt
import configs
from src.datasets.Dataset_DAVIS import Dataset_DAVIS
from src.datasets.Dataset_PESO import Dataset_PESO
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.utils.BaselineConfigs import EXP_OctreeNCA, EXP_OctreeNCA3D, EXP_OctreeNCA3D_superres
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
from src.datasets.png_seg_Dataset import png_seg_Dataset
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
import octree_vis, os, torch, shutil
import pickle as pkl
from src.datasets.Dataset_CholecSeg import Dataset_CholecSeg
from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed

import torchio as tio


print(pc.STUDY_PATH)

study_config = {
    'experiment.name': r'peso_med_fast_dummy',
    'experiment.description': "OctreeNCA2DSegmentation",

    'model.output_channels': 1,
}
study_config = study_config | configs.models.peso_med.peso_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.peso.peso_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config

study_config['experiment.logging.also_eval_on_train'] = False
study_config['experiment.logging.evaluate_interval'] = study_config['trainer.n_epochs']+1
study_config['experiment.task.score'] = ["src.scores.PatchwiseDiceScore.PatchwiseDiceScore",
                                         "src.scores.PatchwiseIoUScore.PatchwiseIoUScore"]


#study_config['experiment.logging.evaluate_interval'] = 1

study_config['trainer.n_epochs'] = 1


study = Study(study_config)


exp = EXP_OctreeNCA().createExperiment(study_config, detail_config={}, dataset_class=Dataset_PESO, dataset_args={
                                                            'patches_path': os.path.join(pc.FILER_BASE_PATH, study_config['experiment.dataset.patches_path']),
                                                            'patch_size': study_config['experiment.dataset.input_size'],
                                                            'path': os.path.join(pc.FILER_BASE_PATH, study_config['experiment.dataset.img_path']),
                                                            'img_level': study_config['experiment.dataset.img_level']
                                                      })
study.add_experiment(exp)

study.run_experiments()
exit()

study.eval_experiments(export_prediction=False)
#figure = octree_vis.visualize(study.experiments[0])

