from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_SAM3D
import octree_vis
from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed


import configs

print("Study Path:", ProjectConfiguration.STUDY_PATH)

study_config = {
    'experiment.name': r'cholec_sam2_base_plus',
    'experiment.description': "Sam3DSegmentation",

    'model.output_channels': 5,
}
study_config = study_config | configs.models.cholec_sam2.cholec_sam2_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.cholec.cholec_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config

study_config['trainer.ema'] = False
study_config['performance.compile'] = False
study_config['experiment.logging.also_eval_on_train'] = False

study_config['trainer.batch_size'] = 1
study_config['experiment.dataset.patch_size'] = [240, 432, 32]

study_config['model.checkpoint'] = "<path>/sam_checkpoints/sam2/sam2_hiera_base_plus.pt"
study_config['model.model_cfg'] = "configs/sam2/sam2_hiera_b+.yaml"

study_config['model.checkpoint'] = "<path>/sam_checkpoints/sam2/sam2_hiera_small.pt"
study_config['model.model_cfg'] = "configs/sam2/sam2_hiera_s.yaml"

study_config['model.checkpoint'] = "<path>/sam_checkpoints/sam2/sam2_hiera_tiny.pt"
study_config['model.model_cfg'] = "configs/sam2/sam2_hiera_t.yaml"


study = Study(study_config)


exp = EXP_SAM3D().createExperiment(study_config, detail_config={}, 
                                            dataset_class=Dataset_CholecSeg_preprocessed, dataset_args = {
                                                'patch_size': study_config['experiment.dataset.patch_size']
                                            })
study.add_experiment(exp)

#study.run_experiments()
study.eval_experiments(pseudo_ensemble=False)
#figure = octree_vis.visualize(study.experiments[0])

#study.eval_experiments_ood()