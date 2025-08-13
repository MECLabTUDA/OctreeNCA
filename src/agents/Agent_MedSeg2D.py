import einops
import torch
from src.agents.Agent import BaseAgent
from src.scores import ScoreList
from src.utils.helper import convert_image, merge_img_label_gt, merge_img_label_gt_simplified
import numpy as np
import math 
from matplotlib import pyplot as plt
import nibabel as nib
import os, pickle as pkl
from src.losses.LossFunctions import DiceLoss
import torchio as tio
import torch.utils.data
import tqdm
from src.utils.ProjectConfiguration import ProjectConfiguration as pc

class Agent_MedSeg2D(BaseAgent):
    @torch.no_grad()
    def test(self, scores: ScoreList, save_img: list = None, tag: str = 'test/img/', 
             pseudo_ensemble: bool = False, 
             split='test', ood_augmentation: tio.Transform|None=None,
             output_name: str=None, export_prediction: bool=False,
             prediction_export_path: str="pred") -> dict:
        r"""Evaluate model on testdata by merging it into 3d volumes first
            TODO: Clean up code and write nicer. Replace fixed images for saving in tensorboard.
            #Args
                dataset (Dataset)
                loss_f (torch.nn.Module)
                steps (int): Number of steps to do for inference
        """
        # Prepare dataset for testing
        dataset = self.exp.datasets[split]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)
        self.exp.set_model_state('test')

        loss_log = {}
        if save_img == None:
            save_img = [1, 2, 3, 4, 5, 32, 45, 89, 357, 53, 122, 267, 97, 389]

        # For each data sample
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = self.prepare_data(data, eval=True)
            patient_id = data['patient_id'][0]
            
            if ood_augmentation != None:
                raise NotImplementedError()
                print(data["image"].shape)
                data['image'] = ood_augmentation(data['image'][0])
                data["image"] = data["image"][None]
                print(data["image"].shape)
                exit()




            image, label = data['image'], data['label']

            #image.shape: BCHW
            #label.shape: BCHW

            if pseudo_ensemble:
                predictions = []
                for k in range(10):
                    predictions.append(self.get_outputs(data, full_img=True, tag=str(k))["logits"])
                stack = torch.stack(predictions, dim=0)
                pred, _ = torch.median(stack, dim=0)
                #nqm_score = self.compute_nqm_score((stack>0).cpu().numpy())
                del predictions, stack
            else:
                pred = self.get_outputs(data, full_img=True, tag="0")["logits"]
            
            pred = einops.rearrange(pred, "b h w c -> b c h w")


            # patchwise scores will update the scores as we continue to iterate
            s: dict = scores(pred=einops.rearrange(pred, "b c h w -> b h w c"), 
                            target=einops.rearrange(label, "b c h w -> b h w c"),
                            patient_id=patient_id)
            
            for key in s.keys():
                if key not in loss_log:
                    loss_log[key] = {}
                loss_log[key][patient_id] = s[key]


            pred = pred.cpu()
            label = label.cpu()
            image = image.cpu()

            if i in save_img and ood_augmentation is None: 
                self.exp.write_img(str(tag) + str(patient_id) + f"_{i}",
                                merge_img_label_gt_simplified(einops.rearrange(image, "b c h w -> b h w c"),
                                                              einops.rearrange(pred, "b c h w -> b h w c"),
                                                               einops.rearrange(label, "b c h w -> b h w c"),
                                                                 dataset.is_rgb),
                                self.exp.currentStep)
                
            if export_prediction:
                exportable_segmentation = einops.rearrange(pred, "b c h w -> b h w c")
                eval_path = os.path.join(pc.FILER_BASE_PATH, self.exp.get_from_config("experiment.model_path"), prediction_export_path)
                os.makedirs(eval_path, exist_ok=True)
                x, y = data['position'][0].item(), data['position'][1].item()
                np.save(os.path.join(eval_path, f"{patient_id}_{x}_{y}"), exportable_segmentation)
        
        ood_label = ""
        if ood_augmentation != None:
            ood_label = str(ood_augmentation)

        print(loss_log)

        # Print dice score per label
        for key in loss_log.keys():
            if len(loss_log[key]) > 0:
                print(f"Average Dice Loss {ood_label} 3d: " + str(key) + ", " + str(sum(loss_log[key].values())/len(loss_log[key])))
                print(f"Standard Deviation {ood_label} 3d: " + str(key) + ", " + str(self.standard_deviation(loss_log[key])))

        self.exp.set_model_state('train')
        return loss_log
    
    def labelVariance(self, images: torch.Tensor, median: torch.Tensor, img_mri: torch.Tensor, img_id: str, targets: torch.Tensor) -> None:
        r"""Calculate variance over all predictions
            #Args
                images (torch): The inferences
                median: The median of all inferences
                img_mri: The mri image
                img_id: The id of the image
                targets: The target segmentation
        """
        mean = np.sum(images, axis=0) / images.shape[0]
        stdd = 0
        for id in range(images.shape[0]):
            img = images[id] - mean
            img = np.power(img, 2)
            stdd = stdd + img
        stdd = stdd / images.shape[0]
        stdd = np.sqrt(stdd)

        print("NQM Score: ", np.sum(stdd) / np.sum(median))
        return stdd