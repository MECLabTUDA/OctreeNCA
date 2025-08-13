import os
from matplotlib import pyplot as plt
import pandas as pd
import torch
import tqdm
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.scores import ScoreList
from src.agents.Agent import BaseAgent
from src.datasets.Dataset_DAVIS import Dataset_DAVIS
from src.utils.helper import convert_image, merge_img_label_gt, merge_img_label_gt_simplified
import numpy as np
import math 
import torch.utils.data
import torchio as tio
import einops

class Agent_MedSeg3D_slicewise(MedNCAAgent):

    @torch.no_grad()
    def test(self, scores: ScoreList, save_img: list = None, tag: str = 'test/img/', 
             pseudo_ensemble: bool = False, 
             split='test', ood_augmentation: tio.Transform|None=None,
             output_name: str=None) -> dict:
        r"""Evaluate model on testdata by merging it into 3d volumes first
            TODO: Clean up code and write nicer. Replace fixed images for saving in tensorboard.
            #Args
                dataset (Dataset)
                loss_f (torch.nn.Module)
                steps (int): Number of steps to do for inference
        """
        # Prepare dataset for testing
        dataset = self.exp.datasets[split]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        self.exp.set_model_state('test')

        # Prepare arrays
        patient_id, patient_3d_image, patient_3d_label = None, [], []
        loss_log = {}

        # For each data sample
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = self.prepare_data(data, eval=True)

            current_patient_id = data['recording_id'][0]
            if patient_id is None or current_patient_id == patient_id:
                patient_3d_image.append(data['image'])
                patient_3d_label.append(data['label'])
                patient_id = current_patient_id
            else:
                loss_log = self.perform_segmentation_main(patient_id, loss_log, scores,dataset.slice, patient_3d_image,
                                                          patient_3d_label,
                                                          pseudo_ensemble, ood_augmentation, dataset.is_rgb,
                                                          tag)


                # start gathering of the new patient MRI
                patient_id = current_patient_id
                patient_3d_image.clear()
                patient_3d_label.clear()
                patient_3d_image.append(data['image'])
                patient_3d_label.append(data['label'])



        #don't forget the very last patient!
        loss_log = self.perform_segmentation_main(patient_id, loss_log, scores,dataset.slice, patient_3d_image,
                                                    patient_3d_label,
                                                    pseudo_ensemble, ood_augmentation, dataset.is_rgb,
                                                    tag)

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
    

    
    def perform_segmentation_slice(self, data: dict, pseudo_ensemble: bool) -> torch.Tensor:
        #image.shape: BCHWD
        if pseudo_ensemble:
            predictions = []
            for i in range(10):
                predictions.append(self.get_outputs(data, full_img=True, tag=str(i))["pred"])
            stack = torch.stack(predictions, dim=0)
            pred, _ = torch.median(stack, dim=0)
            return pred
        else:
            return self.get_outputs(data, full_img=True, tag="0")["pred"]

    def perform_segmentation(self, image: torch.Tensor, pseudo_ensemble: bool)->list[torch.Tensor]:
        slice_preds = []
        for s in range(image.shape[-1]):
            slice_preds.append(self.perform_segmentation_slice({'image': image[...,s],
                                                                'label': None},
                                                                pseudo_ensemble))
            
        return slice_preds


    def perform_segmentation_main(self, patient_id: str,
                                  loss_log: dict,
                                  scores: ScoreList,
                                  dataset_slice: int, 
                                  patient_3d_image: list, 
                                  patient_3d_label: list,
                                  pseudo_ensemble: bool,
                                  ood_augmentation: tio.Transform,
                                  dataset_rgb: bool,
                                  tag: str) -> dict:
        assert dataset_slice == 2, f"{dataset_slice} still needs to be tested"
        patient_3d_image_t = torch.stack(patient_3d_image, dim=dataset_slice+2)#BCHWD
        patient_3d_label_t = torch.stack(patient_3d_label, dim=dataset_slice+2)#BCHWD

        if ood_augmentation != None:
            patient_3d_image_t = ood_augmentation(patient_3d_image_t[0].cpu())[None].to(self.device)

        segmentation = self.perform_segmentation(patient_3d_image_t, pseudo_ensemble)
        patient_3d_prediction_t = torch.stack(segmentation, dim=dataset_slice+1)#BHWDC
        patient_3d_prediction_t = einops.rearrange(patient_3d_prediction_t, "b h w d c -> b c h w d")
        
        #compute scores
        s: dict = scores(einops.rearrange(patient_3d_prediction_t, "b c h w d -> b h w d c"), 
                            einops.rearrange(patient_3d_label_t, "b c h w d -> b h w d c"))
        print(patient_id, ':', s)
        for key in s.keys():
            if key not in loss_log:
                loss_log[key] = {}
            loss_log[key][patient_id] = s[key]

        patient_3d_image_t = patient_3d_image_t.cpu()
        patient_3d_label_t = patient_3d_label_t.cpu()
        patient_3d_prediction_t = patient_3d_prediction_t.cpu()

        # write images to logger
        if ood_augmentation is None:
            self.exp.write_img(str(tag) + str(patient_id),
                            merge_img_label_gt_simplified(patient_3d_image_t, 
                                                            einops.rearrange(patient_3d_prediction_t, 'b c h w d -> b h w d c'), 
                                                            einops.rearrange(patient_3d_label_t, 'b c h w d -> b h w d c'), 
                                                            rgb=dataset_rgb),
                            self.exp.currentStep)

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