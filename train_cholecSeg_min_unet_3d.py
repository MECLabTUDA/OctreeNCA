from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_OctreeNCA3D, EXP_UNet3D, EXP_min_UNet3D
import octree_vis
from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed


import configs

print("Study Path:", ProjectConfiguration.STUDY_PATH)

study_config = {
    'experiment.name': r'cholec_munet3d',
    'experiment.description': "minUNet3DSegmentation",

    'model.output_channels': 5,
}
study_config = study_config | configs.models.min_unet.min_unet_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.cholec.cholec_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config


study_config['model.encoder_name'] = "densenet121"
study_config['experiment.name'] += f"_{study_config['model.arch']}_{study_config['model.encoder_name']}_0"
study_config['model.strides'] = [[2,2,2], [2,2,2], [2,2,2], [2,2,2], [1,1,1]]
study_config['model.encoder_depth'] = 5
study_config['model.decoder_channels'] = [256, 128, 64, 32, 16]
study_config['model.encoder_weights'] = None

study_config['trainer.ema'] = False
study_config['performance.compile'] = False
study_config['experiment.logging.also_eval_on_train'] = False

study_config['trainer.batch_size'] = 2
study_config['experiment.dataset.input_size'] = [256, 448, 80]
study_config['experiment.dataset.patch_size'] = [256, 448, 80]

#study_config["model.eval.patch_wise"] = True # not implemented
#study_config["experiment.logging.evaluate_interval"] = 1


study = Study(study_config)


exp = EXP_min_UNet3D().createExperiment(study_config, detail_config={},
                                            dataset_class=Dataset_CholecSeg_preprocessed, dataset_args = {
                                                'patch_size': study_config['experiment.dataset.patch_size']
                                            })
study.add_experiment(exp)

study.run_experiments()
study.eval_experiments()
#figure = octree_vis.visualize(study.experiments[0])

#study.eval_experiments_ood()