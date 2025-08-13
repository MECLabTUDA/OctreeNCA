from src.datasets.Dataset_PESO import Dataset_PESO
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_OctreeNCA
from src.datasets.Dataset_BCSS_Seg import Dataset_BCSS_Seg
from src.datasets.Dataset_AGGC import Dataset_AGGC
import octree_vis, torch, os, json, openslide, math
import einops
from src.models.Model_OctreeNCAV2 import OctreeNCAV2
import numpy as np, time
import torch.nn.functional as F
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
from src.models.Model_OctreeNCA_2d_patching2 import OctreeNCA2DPatch2
import matplotlib.pyplot as plt
from src.utils.BaselineConfigs import EXP_UNet2D

torch.set_grad_enabled(False)
from src.models.UNetWrapper2D import UNetWrapper2D


model_path = "<path>/Experiments/peso_unet_UNet2DSegmentation/"


with open(os.path.join(model_path, "config.json")) as f:
    config = json.load(f) 

exp = EXP_UNet2D().createExperiment(config, detail_config={}, 
                                                      dataset_class=Dataset_PESO, dataset_args={
                                                            'patches_path': os.path.join(pc.FILER_BASE_PATH, config['experiment.dataset.patches_path']),
                                                            'patch_size': config['experiment.dataset.input_size'],
                                                            'path': os.path.join(pc.FILER_BASE_PATH, config['experiment.dataset.img_path']),
                                                            'img_level': config['experiment.dataset.img_level']
                                                      })

model: UNetWrapper2D = exp.model
assert isinstance(model, UNetWrapper2D)
model = model.eval()
model_device = torch.device("cuda:0")


subject = "14"
pos_x, pos_y = 14400, 24320
size = (180*16, 180*16)
RECORD_MEMORY = False

print(size)


slide = openslide.open_slide(f"<path>/PESO/peso_training/pds_{subject}_HE.tif")
slide = slide.read_region((int(pos_x * slide.level_downsamples[1]),
                        int(pos_y * slide.level_downsamples[1])), 1, size)
slide = np.array(slide)[:,:,0:3]
slide_cpu = slide

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
slide = slide / 255.0
slide = (slide - mean) / std

slide = slide[None]
slide = torch.from_numpy(slide).float().cuda()
slide = einops.rearrange(slide, 'B H W C -> B C H W')

if RECORD_MEMORY:
    torch.cuda.memory._record_memory_history()
torch.cuda.reset_peak_memory_stats()
start_time = time.time()


out = model(slide)['logits']


torch.cuda.synchronize()
end_time = time.time()
print(f"Time: {end_time - start_time}")
if RECORD_MEMORY:
    torch.cuda.memory._dump_snapshot("mem_snapshots/inference2_peso_unet.pickle")
print(torch.cuda.max_memory_allocated() /1000**2)





