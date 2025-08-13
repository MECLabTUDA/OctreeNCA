import imageio, json, os, torch, einops, math, tqdm
import numpy as np
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import time
import nca_cuda3d

from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed
from src.utils.BaselineConfigs import EXP_OctreeNCA3D

torch.set_grad_enabled(False)

from src.models.Model_OctreeNCA_3d_patching2 import OctreeNCA3DPatch2


model_path = "<path>/Experiments/cholec_fast_dummy_M3dSegmentation/"
with open(os.path.join(model_path, "config.json")) as f:
    config = json.load(f) 

exp = EXP_OctreeNCA3D().createExperiment(config, detail_config={}, 
                                        dataset_class=Dataset_CholecSeg_preprocessed, dataset_args = {
                                        })

model: OctreeNCA3DPatch2 = exp.model
assert isinstance(model, OctreeNCA3DPatch2)
model.eval()

def downscale(x: torch.Tensor, out_size):
    x = model.align_tensor_to(x, "BCHWD")
    model.remove_names(x)

    out = F.interpolate(x, size=out_size)
    out.names = ('B', 'C', 'H', 'W', 'D')
    x.names = ('B', 'C', 'H', 'W', 'D')
    return out

video_path = "<path>/Cholec80/cholec80_full_set/videos/video01.mp4"
video_reader = imageio.get_reader(video_path)
n_frames = video_reader.get_meta_data()['duration'] * video_reader.get_meta_data()['fps']

num_seconds = 11
num_frames = int(video_reader.get_meta_data()['fps'] * num_seconds)

num_frames = 18*16
RECORD_MEMORY = False
print(num_frames)




video = []
for frame in range(num_frames):
    image = video_reader.get_data(frame)
    video.append(image[None, ...])

video = np.concatenate(video, axis=0)
video = einops.rearrange(video, 't h w c ->  h w (t c)')


outstacks = []
for i in range(math.ceil(video.shape[-1] / 500)):
    outstack = cv2.resize(video[..., i*500:(i+1)*500], (424, 240))
    outstacks.append(outstack)
video = np.concatenate(outstacks, axis=-1)
video = einops.rearrange(video, 'h w (t c) -> t h w c', c=3).astype(np.float32)
video /= 255.0
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
video -= mean
video /= std
print(video.shape)

video_tensor = torch.from_numpy(einops.rearrange(video, 'D H W C -> 1 H W D C'))
video_tensor.names = ('B', 'H', 'W', 'D', 'C')
computed_resolutions = model.compute_octree_res(video_tensor)
print(computed_resolutions)

seed = torch.zeros(1, *computed_resolutions[-1], model.channel_n,
                                dtype=torch.float, device=model.device, 
                                names=('B', 'H', 'W', 'D', 'C'))
temp = downscale(video_tensor, computed_resolutions[-1])
temp = model.align_tensor_to(temp, "BHWDC")
model.remove_names(temp)
model.remove_names(seed)
seed[:,:,:,:,:model.input_channels] = temp


if RECORD_MEMORY:
    torch.cuda.memory._record_memory_history()
torch.cuda.reset_peak_memory_stats()
start_time = time.time()


state = model.backbone_ncas[1](seed, steps=model.inference_steps[1], fire_rate=model.fire_rate)

state = einops.rearrange(state, '1 H W D C -> 1 C H W D')
state = torch.nn.Upsample(size=computed_resolutions[0], mode='nearest')(state)
temp = F.interpolate(einops.rearrange(video_tensor, "1 h w t c -> 1 c h w t"), size=computed_resolutions[0])
state[0,:model.input_channels,:,:,:] = temp[0]
state = einops.rearrange(state, '1 C H W D -> 1 H W D C')

#state = model.backbone_ncas[0](state, steps=model.inference_steps[0], fire_rate=model.fire_rate)



bb = model.backbone_ncas[0]
state = einops.rearrange(state, "b h w d c -> b c h w d")

const_inputs = state[:,0:bb.input_channels].clone()

for step in range(model.inference_steps[0]):
    # state.shape: BCHW
    delta_state = bb.conv(state)
    delta_state = torch.cat([state, delta_state], dim=1)
    delta_state = bb.fc0(delta_state)
    delta_state = bb.bn(delta_state)
    delta_state = F.relu(delta_state)
    delta_state = bb.fc1(delta_state)

    with torch.no_grad():
        stochastic = torch.zeros([delta_state.size(0),1,
                                delta_state.size(2), delta_state.size(3), delta_state.size(4)], device=delta_state.device)
        stochastic.bernoulli_(p=0.5).float()
    delta_state = delta_state * stochastic

    state = state[:, bb.input_channels:] + delta_state
    state = torch.cat([const_inputs, state], dim=1)

state = einops.rearrange(state, "b c h w d -> b h w d c")


torch.cuda.synchronize()
end_time = time.time()
print(f"Time: {end_time - start_time}")
if RECORD_MEMORY:
    torch.cuda.memory._dump_snapshot("mem_snapshots/inference2_cholec_m3d.pickle")
print(torch.cuda.max_memory_allocated() /1000**2)