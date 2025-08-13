


from matplotlib import pyplot as plt
import numpy as np
from src.datasets import Dataset_Base
import torch
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.models.Model_OctreeNCA import OctreeNCA
from src.models.Model_OctreeNCA_2d_patching2 import OctreeNCA2DPatch2
from src.models.Model_OctreeNCA_3D import OctreeNCA3D
from src.utils.Experiment import Experiment
import torch.utils.data

import torch.utils.data._utils

def find_sample_by_id(experiment: Experiment, dataset: Dataset_Base, sample_id:str):
    try:
        item = dataset.getItemByName(sample_id)
        item = torch.utils.data._utils.collate.default_collate([item])
        return item
    except ValueError:
        assert False, f"Could not find sample with id {sample_id}"

@torch.no_grad()
def visualize(experiment: Experiment, dataset: Dataset_Base = None, split: str='test', sample_id: str=None,
              per_step: bool = False) -> plt.Figure:
    if dataset is None:
        if sample_id is None:
            loader = torch.utils.data.DataLoader(experiment.datasets[split])
            data = next(iter(loader))
        else:
            data = find_sample_by_id(experiment, experiment.datasets[split], sample_id)
    else:
        if sample_id is None:
            loader = torch.utils.data.DataLoader(dataset)
            data = next(iter(loader))
        else:
            data = find_sample_by_id(experiment, dataset, sample_id)


    if isinstance(experiment.model, OctreeNCA2DPatch2):
        return visualize2d(experiment, data, per_step)
    else:
        return visualize3d(experiment, data, per_step)


@torch.no_grad()
def visualize2d(experiment: Experiment, data: dict, per_step: bool) -> plt.Figure:
    assert isinstance(experiment.agent, MedNCAAgent)
    assert isinstance(experiment.model, OctreeNCA2DPatch2)

    data = experiment.agent.prepare_data(data)

    inputs, targets = data['image'], data['label']
    gallery = experiment.model.create_inference_series(inputs, per_step)


    if per_step:
        max_num_steps = max([len(g) for g in gallery])
        num_resolutions = len(gallery)

        SIZE_PER_SUBPLOT = 2

        figure = plt.figure(figsize=(max_num_steps * SIZE_PER_SUBPLOT, num_resolutions * SIZE_PER_SUBPLOT))

        for resolution_idx in range(num_resolutions):
            for step_idx, img in enumerate(gallery[resolution_idx]):
                plt.subplot(num_resolutions, max_num_steps, resolution_idx*max_num_steps + step_idx + 1)
                min_max_normalize = False
                if min_max_normalize:
                    _img = img[0, :, :, 3].cpu().numpy()
                    _img = (_img - _img.min()) / (_img.max() - _img.min() + 0.0001)
                    plt.imshow(_img, vmin=0, vmax=1, cmap='PiYG')
                else:
                    _img = img[0, :, :, 3].cpu()
                    _img = torch.sigmoid(_img).numpy()
                    plt.imshow(_img, vmin=0, vmax=1, cmap='PiYG')
                plt.axis('off')
                if step_idx == 0:
                    plt.title(f"{img.shape[1]}x{img.shape[2]}", fontsize=8)
    else:
        #convert to binary label
        prediction = (gallery[-1] > 0).float()
        

        figure = plt.figure(figsize=(25, 5))
        #plot all figures in gallery
        for i, img in enumerate(gallery):
            plt.subplot(1, len(gallery)+2, i+1)
            min_max_normalize = False
            if min_max_normalize:
                _img = img[0, :, :, 3].cpu().numpy()
                _img = (_img - _img.min()) / (_img.max() - _img.min() + 0.0001)
                plt.imshow(_img, vmin=0, vmax=1, cmap='PiYG')
            else:
                _img = img[0, :, :, 3].cpu()
                _img = torch.sigmoid(_img).numpy()
                plt.imshow(_img, vmin=0, vmax=1, cmap='PiYG')
            plt.title(f"{img.shape[1]}x{img.shape[2]}", fontsize=8)
            plt.axis('off')
            
        plt.subplot(1, len(gallery)+2, i+2)
        plt.imshow(prediction[0, :, :, 3].cpu().numpy())
        plt.title(f"prediction", fontsize=8)
        plt.axis('off')

        plt.subplot(1, len(gallery)+2, i+3)
        plt.imshow(targets[0, 0, :, :].cpu().numpy())
        plt.title(f"ground truth", fontsize=8)
        plt.axis('off')

    plt.savefig("inference_test.png", bbox_inches='tight')
    return figure


@torch.no_grad()
def visualize3d(experiment: Experiment, data: dict, per_step: bool) -> plt.Figure:
    assert isinstance(experiment.agent, M3DNCAAgent)
    assert isinstance(experiment.model, OctreeNCA3D)

    experiment.agent.prepare_data(data)
    inputs, targets = data['image'], data['label']
    gallery = experiment.model.create_inference_series(inputs) #list of BHWDC tensors

    #convert to binary label
    gallery.append((gallery[-1] > 0).float())
    
    figure = plt.figure(figsize=(15, 5))
    #plot all figures in gallery
    for i, img in enumerate(gallery):
        depth = img.shape[3]
        plt.subplot(1, len(gallery)+1, i+1)
        plt.imshow(img[0, :, :, depth//2, 1].cpu().numpy())
        plt.title(f"{img.shape[1]}x{img.shape[2]}", fontsize=8)
        plt.axis('off')
        
    plt.subplot(1, len(gallery)+1, i+2)
    #targets: BHWDC
    plt.imshow(targets[0, :, :, depth//2, 0].cpu().numpy())
    plt.title(f"ground truth", fontsize=8)
    plt.axis('off')
    return figure