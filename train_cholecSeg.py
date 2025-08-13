from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_OctreeNCA3D
import octree_vis
from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed
import configs

print("Study Path:", ProjectConfiguration.STUDY_PATH)

study_config = {
    'experiment.name': r'cholecGN',
    'experiment.description': "OctreeNCASegmentation",

    'model.output_channels': 5,
}
study_config = study_config | configs.models.cholec.cholec_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.cholec.cholec_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config

study_config['performance.compile'] = False
study_config['experiment.logging.also_eval_on_train'] = False

study_config['model.normalization'] = "group"


study_config['model.kernel_size'] = [3, 3, 3, 3, 3]





study_config['model.normalization'] = "none"    #"none"

steps = 10                                      # 10
alpha = 1.0                                     # 1.0
study_config['model.octree.res_and_steps'] = [[[240, 432, 80], steps], [[120, 216, 40], steps], [[60,108,20], steps], [[30,54,10], steps], [[15,27,5], int(alpha * 27)]]


study_config['model.channel_n'] = 16            # 16
study_config['model.hidden_size'] = 64          # 64

study_config['trainer.batch_size'] = 3          # 3

dice_loss_weight = 1.0                          # 1.0


ema_decay = 0.99                               # 0.99
study_config['trainer.ema'] = ema_decay > 0.0
study_config['trainer.ema.decay'] = ema_decay


study_config['trainer.losses'] = ["src.losses.DiceLoss.DiceLoss", "src.losses.BCELoss.BCELoss"]
study_config['trainer.losses.parameters'] = [{}, {}]
study_config['trainer.loss_weights'] = [dice_loss_weight, 2.0-dice_loss_weight]
#study_config['trainer.loss_weights'] = [1.5, 0.5]

study_config['experiment.name'] = f"cholecfFixAbl_{study_config['model.normalization']}_{steps}_{alpha}_{study_config['model.channel_n']}_{study_config['trainer.batch_size']}_{dice_loss_weight}_{ema_decay}"



study_config['experiment.name'] = "TEST1"
study_config['trainer.n_epochs'] = 1
study_config['trainer.num_steps_per_epoch'] = 1
study_config['experiment.save_interval'] = 1

study_config['experiment.task.score'] = ["src.scores.DiceScore.DiceScore", "src.scores.IoUScore.IoUScore"]




study = Study(study_config)

exp = EXP_OctreeNCA3D().createExperiment(study_config, detail_config={}, 
                                            dataset_class=Dataset_CholecSeg_preprocessed, dataset_args = {
                                            })
study.add_experiment(exp)

study.run_experiments()
study.eval_experiments()
#figure = octree_vis.visualize(study.experiments[0])
