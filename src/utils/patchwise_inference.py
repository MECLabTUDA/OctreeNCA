import einops
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.models.Model_OctreeNCA_3d_patching2 import OctreeNCA3DPatch2
@torch.no_grad()
def patchwise_inference3d(inputs: torch.Tensor, model : OctreeNCA3DPatch2, patch_sizes) -> torch.Tensor:
    assert isinstance(model, OctreeNCA3DPatch2)
    computed_octree_res = model.compute_octree_res(inputs)
    if not model.separate_models:
        models = [model.backbone_nca] * len(patch_sizes)
    else:
        models = model.backbone_ncas

    state = torch.zeros(1, *computed_octree_res[-1], model.channel_n, 
                        device=model.device)
    inputs = einops.rearrange(inputs, 'b x y z c -> b c x y z')
    temp = F.interpolate(inputs, size=computed_octree_res[-1], mode='trilinear')
    temp = einops.rearrange(temp, 'b c x y z -> b x y z c')
    state[:,:,:,:,:model.input_channels] = temp

    for level, patch in reversed(list(enumerate(patch_sizes))):
        if patch is None:
            state = models[level](state, steps=model.inference_steps[level], 
                            fire_rate=model.fire_rate)
        else:
            new_state = torch.zeros(state.shape[0], state.shape[1], state.shape[2], state.shape[3], state.shape[4], device=model.device)
            for x in range(0, state.shape[1], patch[0]):
                for y in range(0, state.shape[2], patch[1]):
                    for z in range(0, state.shape[3], patch[2]):
                        padding = model.inference_steps[level]
                        write_start_x = max(x,0)
                        write_start_y = max(y,0)
                        write_start_z = max(z,0)
                        write_end_x = min(x+patch[0], state.shape[1])
                        write_end_y = min(y+patch[1], state.shape[2])
                        write_end_z = min(z+patch[2], state.shape[3])
                        read_start_x = max(x-padding,0)
                        read_start_y = max(y-padding,0)
                        read_start_z = max(z-padding,0)
                        read_end_x = min(x+patch[0]+padding, state.shape[1])
                        read_end_y = min(y+patch[1]+padding, state.shape[2])
                        read_end_z = min(z+patch[2]+padding, state.shape[3])
                        padding_start_x = write_start_x - read_start_x
                        padding_start_y = write_start_y - read_start_y
                        padding_start_z = write_start_z - read_start_z
                        padding_end_x = read_end_x - write_end_x
                        padding_end_y = read_end_y - write_end_y
                        padding_end_z = read_end_z - write_end_z

                        temp_state = state[:, read_start_x:read_end_x, read_start_y:read_end_y, read_start_z:read_end_z, :]
                        temp_state = models[level](temp_state, steps=model.inference_steps[level],
                                            fire_rate=model.fire_rate)
                        new_state[:, write_start_x:write_end_x, 
                                    write_start_y:write_end_y, 
                                    write_start_z:write_end_z, 
                                    :] = temp_state[:, 
                                                    padding_start_x:patch[0]+padding_start_x, 
                                                    padding_start_y:patch[1]+padding_start_y, 
                                                    padding_start_z:patch[2]+padding_start_z, :]
            state = new_state
        
        if level > 0:
            state = einops.rearrange(state, 'b x y z c -> b c x y z')
            state = F.interpolate(state, size=computed_octree_res[level-1], mode='trilinear')
            state = einops.rearrange(state, 'b c x y z -> b x y z c')
            temp = F.interpolate(inputs, size=computed_octree_res[level-1], mode='trilinear')
            temp = einops.rearrange(temp, 'b c x y z -> b x y z c')
            state[:,:,:,:,:model.input_channels] = temp
    return state