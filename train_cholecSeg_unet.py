from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_OctreeNCA3D, EXP_UNet3D
import octree_vis
from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed


import configs

print("Study Path:", ProjectConfiguration.STUDY_PATH)

study_config = {
    'experiment.name': r'cholec_unet',
    'experiment.description': "UNetSegmentation",

    'model.output_channels': 5,
}
study_config = study_config | configs.models.cholec_unet.cholec_unet_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.cholec.cholec_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config

study_config['trainer.ema'] = False
study_config['performance.compile'] = False
study_config['experiment.logging.also_eval_on_train'] = False

study_config['trainer.batch_size'] = 1
study_config['experiment.dataset.patch_size'] = [240, 432, 32]

study = Study(study_config)

##23 GB roughly!

exp = EXP_UNet3D().createExperiment(study_config, detail_config={}, 
                                            dataset_class=Dataset_CholecSeg_preprocessed, dataset_args = {
                                                'patch_size': study_config['experiment.dataset.patch_size']
                                            })
study.add_experiment(exp)

study.run_experiments()
study.eval_experiments()