from src.datasets.Dataset_PESO import Dataset_PESO
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_OctreeNCA
from src.datasets.Dataset_BCSS_Seg import Dataset_BCSS_Seg
from src.datasets.Dataset_AGGC import Dataset_AGGC
import octree_vis, torch, os, json, openslide, math
import einops
from src.models.Model_OctreeNCAV2 import OctreeNCAV2
import numpy as np
import torch.nn.functional as F
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
from src.models.Model_OctreeNCA_2d_patching2 import OctreeNCA2DPatch2
import matplotlib.pyplot as plt
import time
torch.set_grad_enabled(False)

model_path = "<path>/Experiments/pesofAbl_none_10_1.5_16_3_1.0_0.99_OctreeNCA2DSegmentation"

with open(os.path.join(model_path, "config.json")) as f:
    config = json.load(f) 

exp = EXP_OctreeNCA().createExperiment(config, detail_config={}, 
                                                      dataset_class=Dataset_PESO, dataset_args={
                                                            'patches_path': os.path.join(pc.FILER_BASE_PATH, config['experiment.dataset.patches_path']),
                                                            'patch_size': config['experiment.dataset.input_size'],
                                                            'path': os.path.join(pc.FILER_BASE_PATH, config['experiment.dataset.img_path']),
                                                            'img_level': config['experiment.dataset.img_level']
                                                      })

model: OctreeNCA2DPatch2 = exp.model
assert isinstance(model, OctreeNCA2DPatch2)
model = model.eval()

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


def load_sample(size):
    subject = "14"
    pos_x, pos_y = 14400, 24320
    #pos_x, pos_y = 14400 - 1000, 24320
    #size = (16*161, 16*161)
    #size = (320, 320)

    slide = openslide.open_slide(f"<path>/PESO/peso_training/pds_{subject}_HE.tif")
    slide = slide.read_region((int(pos_x * slide.level_downsamples[1]),
                            int(pos_y * slide.level_downsamples[1])), 1, size)
    #slide = slide.read_region((int(pos_x * slide.level_downsamples[1]),
    #                           int(pos_y * slide.level_downsamples[1])), 1, (16*10, 16*10))
    slide = np.array(slide)[:,:,0:3]
    slide_cpu = slide

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    slide = slide / 255.0
    slide = (slide - mean) / std

    slide = slide[None]
    slide = torch.from_numpy(slide).float()
    slide.names = ('B', 'H', 'W', 'C')

    slide = align_tensor_to(slide, "BHWC")
    computed_resolutions = compute_resolutions(slide.shape[1:3], model)
    print(computed_resolutions)

    seed = torch.zeros(1, *computed_resolutions[-1], model.channel_n,
                                    dtype=torch.float, device=slide.device,
                                    names=('B', 'H', 'W', 'C'))
    temp = downscale(slide, computed_resolutions[-1])
    temp = align_tensor_to(temp, "BHWC")
    remove_names(temp)
    remove_names(seed)
    seed[:,:,:,:model.input_channels] = temp
    #temp.names = ('B', 'H', 'W', 'C')
    #seed.names = ('B', 'H', 'W', 'C')

    return seed, slide, computed_resolutions



def perform_inference(seed, slide, computed_resolutions):
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


def measure_vram_and_print(size):
    seed, slide, computed_resolutions = load_sample(size)
    seed = seed.cuda()
    slide = slide.cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(model.device)
    print("about to start inference")
    perform_inference(seed, slide, computed_resolutions)
    mem_allocation = torch.cuda.max_memory_allocated(model.device)
    mem_allocation_mb = mem_allocation/ 1024**2
    print("Max VRAM:", mem_allocation_mb, "MB")
    return mem_allocation_mb



num_pixel_to_measurements = {}
for mult in [20, 40, 80, 160, 300, 480][::-1]:
    size = (mult*16, mult*16)
    num_pixel = size[0] * size[1]
    num_pixel_to_measurements[num_pixel] = []
    for _ in range(3):
        num_pixel_to_measurements[num_pixel].append(measure_vram_and_print(size))


json.dump(num_pixel_to_measurements, open("inference_pathology_oct_gpu_vram.json", "w"), indent=4)