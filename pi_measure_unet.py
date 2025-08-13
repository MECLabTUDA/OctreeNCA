import configs, torch
from src.models.Model_OctreeNCA_2d_patching2 import OctreeNCA2DPatch2
import time, json
from src.models.UNetWrapper2D import UNetWrapper2D
from unet import UNet2D


torch.set_grad_enabled(False)

study_config = {
    'experiment.name': r'pesoXXXS',
    'experiment.description': "UNet2DSegmentation",

    'model.output_channels': 1,
}
study_config = study_config | configs.models.peso_unet.peso_unet_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.peso.peso_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config

study_config['experiment.logging.also_eval_on_train'] = False
study_config['experiment.logging.evaluate_interval'] = study_config['trainer.n_epochs']+1
study_config['experiment.task.score'] = ["src.scores.PatchwiseDiceScore.PatchwiseDiceScore",
                                         "src.scores.PatchwiseIoUScore.PatchwiseIoUScore"]


study_config['trainer.ema'] = False
study_config['trainer.batch_size'] = 10



#study_config['model.num_encoding_blocks'] = 3
#study_config['model.out_channels_first_layer'] = 8



study_config['experiment.device'] = "cpu"

config = study_config

model_params = {k.replace("model.", ""): v for k, v in config.items() if k.startswith('model.')}
model_params.pop("output_channels")
model_params.pop("input_channels")
model = UNet2D(in_channels=config['model.input_channels'], out_classes=config['model.output_channels'], padding=1, **model_params)
model = UNetWrapper2D(model).eval()


def perform_inference_and_measure_time(img_dim):
    input_img = torch.rand(1, 3, img_dim, img_dim)  #this must be BCHW
    dummy_seg = torch.rand(1, 3, img_dim, img_dim)  #this must be BCHW
    start = time.time()
    out = model(input_img, dummy_seg)
    end = time.time()
    assert out['logits'].shape[1:3] == (img_dim, img_dim)
    return end-start


results = {}
for img_dim in [320, 320*2, 320*3, 320*4, 320*5]:
    print(img_dim)
    timings = []
    for i in range(3):
        print("run", i)
        timings.append(perform_inference_and_measure_time(img_dim))

    results[img_dim] = timings
    with open("pi_timing_results_unet_pi.json", "w") as f:
        json.dump(results, f, indent=4)