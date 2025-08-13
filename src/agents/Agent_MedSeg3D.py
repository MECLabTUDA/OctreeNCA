import os
import einops
from matplotlib import pyplot as plt
import pandas as pd
import torch
from src.scores import ScoreList
from src.agents.Agent import BaseAgent
from src.datasets.Dataset_DAVIS import Dataset_DAVIS
from src.utils.helper import convert_image, merge_img_label_gt, merge_img_label_gt_simplified
import numpy as np
import math 
import torch.utils.data
import torchio as tio

class Agent_MedSeg3D(BaseAgent):

    @torch.no_grad()
    def test(self, scores: ScoreList, save_img: list = None, tag: str = 'test/img/', 
             pseudo_ensemble: bool = False, 
             split='test', ood_augmentation: tio.Transform|None=None,
             output_name: str=None,
             export_prediction: bool = False) -> dict:
        r"""Evaluate model on testdata by merging it into 3d volumes first
            TODO: Clean up code and write nicer. Replace fixed images for saving in tensorboard.
            #Args
                dataset (Dataset)
                loss_f (torch.nn.Module)
                steps (int): Number of steps to do for inference
        """
        dataset = self.exp.datasets[split]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        self.exp.set_model_state('test')
        
        loss_log = {}

        # For each data sample
        for i, data in enumerate(dataloader):
            data = self.prepare_data(data, eval=True)
            assert data['image'].shape[0] == 1, "Batch size must be 1 for evaluation"

            if ood_augmentation != None:
                data['image'] = ood_augmentation(data['image'][0].cpu()).to(self.device)
                data["image"] = data["image"][None]

            patient_id, image, label = data['recording_id'][0], data['image'], data['label']
            recording_id = data['recording_id'][0]

            # image.shape: BCHWD
            label = einops.rearrange(label, "b h w d c -> b c h w d")

            if pseudo_ensemble:
                predictions = []
                for i in range(10):
                    predictions.append(self.get_outputs(data, full_img=True, tag=str(i))["logits"])
                stack = torch.stack(predictions, dim=0)
                pred, _ = torch.median(stack, dim=0)
                nqm_score = self.compute_nqm_score((stack>0).cpu().numpy())
                del predictions, stack
            else:
                pred = self.get_outputs(data, full_img=True, tag="0")["logits"]

            pred = einops.rearrange(pred, "b h w d c -> b c h w d")


            s: dict = scores(einops.rearrange(pred, "b c h w d -> b h w d c"), 
                            einops.rearrange(label, "b c h w d -> b h w d c"))
            
            if pseudo_ensemble:
                s["NQMScore"] = nqm_score

            print(recording_id, ':', s)
            for key in s.keys():
                if key not in loss_log:
                    loss_log[key] = {}
                loss_log[key][recording_id] = s[key]


            pred = pred.cpu()
            label = label.cpu()
            image = image.cpu()

            # write images to logger
            if ood_augmentation is None:
                self.exp.write_img(str(tag) + str(recording_id),
                                merge_img_label_gt_simplified(image, 
                                                                einops.rearrange(pred, 'b c h w d -> b h w d c'), 
                                                                einops.rearrange(label, 'b c h w d -> b h w d c'), 
                                                                rgb=dataset.is_rgb),
                                self.exp.currentStep)
            if export_prediction:
                exportable_segmentation = einops.rearrange(pred, "b c h w d -> b h w d c")[0]
                eval_path = os.path.join(self.exp.get_from_config('experiment.output_path'), 'pred')
                os.makedirs(eval_path, exist_ok=True)
                np.save(os.path.join(eval_path, f"{recording_id}.npy"), exportable_segmentation.numpy())
        
        ood_label = ""
        if ood_augmentation != None:
            ood_label = str(ood_augmentation)

        # Print dice score per label
        for key in loss_log.keys():
            if len(loss_log[key]) > 0:
                print(f"Average Dice Loss {ood_label} 3d: " + str(key) + ", " + str(sum(loss_log[key].values())/len(loss_log[key])))
                print(f"Standard Deviation {ood_label} 3d: " + str(key) + ", " + str(self.standard_deviation(loss_log[key])))

        self.exp.set_model_state('train')


        return loss_log