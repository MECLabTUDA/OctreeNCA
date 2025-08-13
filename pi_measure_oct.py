import configs, torch
from src.models.Model_OctreeNCA_2d_patching2 import OctreeNCA2DPatch2
import time, json, einops
import torch.nn.functional as F
import math, numpy as np

torch.set_grad_enabled(False)

study_config = {
    'experiment.name': r'pesoS10NN',
    'experiment.description': "OctreeNCA2DSegmentation",

    'model.output_channels': 1,
}
study_config = study_config | configs.models.peso.peso_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.peso.peso_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config

study_config['experiment.logging.also_eval_on_train'] = False
study_config['experiment.logging.evaluate_interval'] = study_config['trainer.n_epochs']+1
study_config['experiment.task.score'] = ["src.scores.PatchwiseDiceScore.PatchwiseDiceScore",
                                         "src.scores.PatchwiseIoUScore.PatchwiseIoUScore"]

study_config['model.normalization'] = "none"    #"none"

steps = 10                                      # 10
alpha = 1.0                                     # 1.0
study_config['model.octree.res_and_steps'] = [[[320,320], steps], [[160,160], steps], [[80,80], steps], [[40,40], steps], [[20,20], int(alpha * 20)]]


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

study_config['experiment.name'] = f"pesofAbl_{study_config['model.normalization']}_{steps}_{alpha}_{study_config['model.channel_n']}_{study_config['trainer.batch_size']}_{dice_loss_weight}_{ema_decay}"

study_config['experiment.device'] = "cpu"



assert study_config['model.backbone_class'] == "BasicNCA2DFast"
model = OctreeNCA2DPatch2(study_config).eval()

def compute_resolutions(x_shape, model):
    upscale_factors = []
    for i in range(len(model.octree_res)-1):
        t = []
        for c in range(2):
            t.append(model.octree_res[i][c]//model.octree_res[i+1][c])
        upscale_factors.append(t)

    new_octree_res = [tuple(x_shape)]
    for i in range(1, len(model.octree_res)):
        downsample_factor = np.array(model.octree_res[i-1]) / np.array(model.octree_res[i])
        new_octree_res.append([math.ceil(new_octree_res[i-1][0] / downsample_factor[0]), 
                                math.ceil(new_octree_res[i-1][1] / downsample_factor[1])])
    return new_octree_res

def remove_names(x: torch.Tensor):
    x.names = [None] * len(x.names)
    return x

def align_tensor_to(x: torch.Tensor, target: str):
    if isinstance(target, tuple):
        target_str = ' '.join(target)
    elif isinstance(target, str): 
        if max(map(len, target.split())) != 1:
            #targets are like "BCHW"
            target_str = ' '.join(target)
        else:
            #targets are like "B C H W"
            target_str = target
            target = target.replace(" ", "")


    pattern = f"{' '.join(x.names)} -> {target_str}"
    x = remove_names(x)
    x = einops.rearrange(x, pattern)
    x.names = tuple(target)
    return x

def downscale(x: torch.Tensor, out_size):
    x = align_tensor_to(x, "BCHW")
    remove_names(x)

    out = F.interpolate(x, size=out_size)
    out.names = ('B', 'C', 'H', 'W')
    x.names = ('B', 'C', 'H', 'W')
    return out



def perform_inference(slide, computed_resolutions):
    seed = torch.zeros(1, *computed_resolutions[-1], model.channel_n,
                                    dtype=torch.float, device=slide.device,
                                    names=('B', 'H', 'W', 'C'))
    temp = downscale(slide, computed_resolutions[-1])
    temp = align_tensor_to(temp, "BHWC")
    remove_names(temp)
    remove_names(seed)
    slide = align_tensor_to(slide, "BHWC")
    remove_names(slide)
    seed[:,:,:,:model.input_channels] = temp

    state = model.backbone_ncas[4](seed, steps=model.inference_steps[4], fire_rate=model.fire_rate)

    state = einops.rearrange(state, "B H W C -> B C H W")
    state = torch.nn.Upsample(size=computed_resolutions[3], mode='nearest')(state)
    temp = F.interpolate(einops.rearrange(slide, "B H W C -> B C H W"), size=computed_resolutions[3])
    state[0,:model.input_channels,:,:] = temp[0]
    state = einops.rearrange(state, "B C H W -> B H W C")
    state = model.backbone_ncas[3](state, steps=model.inference_steps[3], fire_rate=model.fire_rate)


    state = einops.rearrange(state, "B H W C -> B C H W")
    state = torch.nn.Upsample(size=computed_resolutions[2], mode='nearest')(state)
    temp = F.interpolate(einops.rearrange(slide, "B H W C -> B C H W"), size=computed_resolutions[2])
    state[0,:model.input_channels,:,:] = temp[0]
    state = einops.rearrange(state, "B C H W -> B H W C")
    state = model.backbone_ncas[2](state, steps=model.inference_steps[3], fire_rate=model.fire_rate)


    state = einops.rearrange(state, "B H W C -> B C H W")
    state = torch.nn.Upsample(size=computed_resolutions[2], mode='nearest')(state)
    temp = F.interpolate(einops.rearrange(slide, "B H W C -> B C H W"), size=computed_resolutions[2])
    state[0,:model.input_channels,:,:] = temp[0]
    state = einops.rearrange(state, "B C H W -> B H W C")
    state = model.backbone_ncas[2](state, steps=model.inference_steps[2], fire_rate=model.fire_rate)


    state = einops.rearrange(state, "B H W C -> B C H W")
    state = torch.nn.Upsample(size=computed_resolutions[1], mode='nearest')(state)
    temp = F.interpolate(einops.rearrange(slide, "B H W C -> B C H W"), size=computed_resolutions[1])
    state[0,:model.input_channels,:,:] = temp[0]
    state = einops.rearrange(state, "B C H W -> B H W C")
    state = model.backbone_ncas[1](state, steps=model.inference_steps[1], fire_rate=model.fire_rate)


    state = einops.rearrange(state, "B H W C -> B C H W")
    state = torch.nn.Upsample(size=computed_resolutions[0], mode='nearest')(state)
    temp = F.interpolate(einops.rearrange(slide, "B H W C -> B C H W"), size=computed_resolutions[0])
    state[0,:model.input_channels,:,:] = temp[0]
    state = einops.rearrange(state, "B C H W -> B H W C")
    state = model.backbone_ncas[0](state, steps=model.inference_steps[0], fire_rate=model.fire_rate)


def perform_inference_and_measure_time(img_dim):
    input_img = torch.rand(1, 3, img_dim, img_dim, names=('B', 'C', 'H', 'W'))  #this must be BCHW
    computed_resolutions = compute_resolutions(input_img.shape[2:], model)
    start = time.time()
    perform_inference(input_img, computed_resolutions)
    end = time.time()
    return end-start

results = {}
for img_dim in [320, 320*2, 320*3, 320*4, 320*5]:
    print(img_dim)
    timings = []
    for i in range(3):
        print("run", i)
        timings.append(perform_inference_and_measure_time(img_dim))

    results[img_dim] = timings
    with open("pi_timing_results_oct_pi.json", "w") as f:
        json.dump(results, f, indent=4)