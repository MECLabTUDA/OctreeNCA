import imageio, json, os, torch, einops, math, tqdm
import numpy as np
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import time
import nca_cuda3d
import colormaps as cmaps

from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed
from src.utils.BaselineConfigs import EXP_OctreeNCA3D

torch.set_grad_enabled(False)

from src.models.Model_OctreeNCA_3d_patching2 import OctreeNCA3DPatch2


model_path = "<path>/Experiments/cholecfFixAbl_none_10_1.0_16_1_1.0_0.99_OctreeNCASegmentation/"
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

start = 120
#num_frames = 119*16
num_frames = 18*16
RECORD_MEMORY = False
CUDA = False
EXPORT_VIDEO = True
EXPORT_AS_ARRAY = False
print(num_frames)

if CUDA:
    for bb in model.backbone_ncas:
        bb.use_forward_cuda = True



video = []
for frame in range(start, start+num_frames):
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


state = model.backbone_ncas[4](seed, steps=model.inference_steps[4], fire_rate=model.fire_rate)

state = einops.rearrange(state, '1 H W D C -> 1 C H W D')
state = torch.nn.Upsample(size=computed_resolutions[3], mode='nearest')(state)
temp = F.interpolate(einops.rearrange(video_tensor, "1 h w t c -> 1 c h w t"), size=computed_resolutions[3])
state[0,:model.input_channels,:,:,:] = temp[0]
state = einops.rearrange(state, '1 C H W D -> 1 H W D C')
state = model.backbone_ncas[3](state, steps=model.inference_steps[3], fire_rate=model.fire_rate)

state = einops.rearrange(state, '1 H W D C -> 1 C H W D')
state = torch.nn.Upsample(size=computed_resolutions[2], mode='nearest')(state)
temp = F.interpolate(einops.rearrange(video_tensor, "1 h w t c -> 1 c h w t"), size=computed_resolutions[2])
state[0,:model.input_channels,:,:,:] = temp[0]
state = einops.rearrange(state, '1 C H W D -> 1 H W D C')
state = model.backbone_ncas[2](state, steps=model.inference_steps[2], fire_rate=model.fire_rate)

state = einops.rearrange(state, '1 H W D C -> 1 C H W D')
state = torch.nn.Upsample(size=computed_resolutions[1], mode='nearest')(state)
temp = F.interpolate(einops.rearrange(video_tensor, "1 h w t c -> 1 c h w t"), size=computed_resolutions[1])
state[0,:model.input_channels,:,:,:] = temp[0]
state = einops.rearrange(state, '1 C H W D -> 1 H W D C')
state = model.backbone_ncas[1](state, steps=model.inference_steps[1], fire_rate=model.fire_rate)

state = einops.rearrange(state, '1 H W D C -> 1 C H W D')
if computed_resolutions[0][2] > 1000:
    MAX_DEPTH = 37*16
    new_state = torch.zeros(1, state.size(1), *computed_resolutions[0], device=state.device)
    for c in range(0, state.size(1)):
        new_state[:, c:c+1] = torch.nn.Upsample(size=computed_resolutions[0], mode='nearest')(state[:, c:c+1])
    state = new_state
    del new_state
    assert state.shape[2:] == tuple(computed_resolutions[0]), f"{state.shape} != {computed_resolutions[0]}"
else:
    state = torch.nn.Upsample(size=computed_resolutions[0], mode='nearest')(state)
temp = F.interpolate(einops.rearrange(video_tensor, "1 h w t c -> 1 c h w t"), size=computed_resolutions[0])
state[0,:model.input_channels,:,:,:] = temp[0]
state = einops.rearrange(state, '1 C H W D -> 1 H W D C')

#state = model.backbone_ncas[0](state, steps=model.inference_steps[0], fire_rate=model.fire_rate)


if CUDA:
    bb = model.backbone_ncas[0]
    state = einops.rearrange(state, "b h w d c -> b c h w d")
    state = state.contiguous()
    
    state2 = torch.zeros(state.size(0), state.size(1), state.size(2), state.size(3), state.size(4), device=state.device)
    for step in range(model.inference_steps[0]):
        if step % 2 == 0:
            state2[:, 0].bernoulli_(0.5)
            nca_cuda3d.nca3d_cuda(state2, state, bb.conv.weight, bb.conv.bias, bb.fc0.weight, bb.fc0.bias, bb.fc1.weight)
        else:
            state[:, 0].bernoulli_(0.5)
            nca_cuda3d.nca3d_cuda(state, state2, bb.conv.weight, bb.conv.bias, bb.fc0.weight, bb.fc0.bias, bb.fc1.weight)

    if model.inference_steps[0] % 2 == 0:
        del state2
    else:
        del state
        state = state2

    state = einops.rearrange(state, "b c h w d -> b h w d c")

else:
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
    if CUDA:
        torch.cuda.memory._dump_snapshot("mem_snapshots/inference2_cholec_oct_cuda.pickle")
    else:
        torch.cuda.memory._dump_snapshot("mem_snapshots/inference2_cholec_oct.pickle")
print(torch.cuda.max_memory_allocated() /1000**2)

state = state[..., model.input_channels:]
state = state[..., :model.output_channels]
segmentation = state.cpu().numpy() > 0
if EXPORT_VIDEO:
    color_dict={
        0: (127, 60, 141), 
        1: (17, 165, 121), 
        2: (57, 105, 172),
        3: (242, 183, 1),
        4: (231, 63, 116),
    }
    video_reader = imageio.get_reader(video_path)
    video = []
    for frame in range(start, start+num_frames):
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

    if CUDA:
        out_path = "<path>/qualitative/inference2_cholec_oct_cuda.mp4"
    else:
        out_path = "<path>/qualitative/inference2_cholec_oct.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (424, 240), True)
    for i in range(segmentation.shape[3]):
        frame_seg = segmentation[0, :, :, i, :]
        frame = np.zeros((segmentation.shape[1], segmentation.shape[2], 3), dtype=np.uint8)
        mask = np.zeros((segmentation.shape[1], segmentation.shape[2]), dtype=bool)
        for c in range(model.output_channels):
            frame[frame_seg[..., c], :] = cmaps.bold[c].colors[::-1] * 255
            mask[frame_seg[..., c]] = True


        frame[mask] = 0.5 * frame[mask] + 0.5 * video[i, mask]
        frame[~mask] = video[i, ~mask]
        
        frame = frame.astype(np.uint8)
        out.write(frame)

    out.release()

if EXPORT_AS_ARRAY:
    seg_array = segmentation[:, :, :, :, :model.output_channels]
    np.save("<path>/qualitative/inference2_cholec_oct.npy", seg_array)