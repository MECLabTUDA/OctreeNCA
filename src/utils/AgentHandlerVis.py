from typing import List, Dict, Any, Callable
import torch
import torchio
from PIL import Image
from os.path import join
import numpy as np
import copy
from src.datasets.Dataset_Base import Dataset_Base
from src.agents.Agent import BaseAgent
from src.utils.Experiment_vis import Experiment_vis
from src.utils.helper import merge_img_label_gt
from src.utils.DatasetWrapperVis import DatasetWrapperVis
from src.utils.BasicNCA3DVis import BasicNCA3DVis
"""
DEPRECATED
DEPRECATED
DEPRECATED
DEPRECATED
DO NOT USE DO NOT USE DO NOT USE

"""
class AgentHandlerVis3D():
    """
    This class is intended as a wrapper for an Agent 
    for visualization purposes.
    """
    dataset: Dataset_Base
    dataset_wrapper: DatasetWrapperVis
    agent: BaseAgent
    config: List[Dict[str, Any]]
    device: torch.device
    def __init__(self, dataset: Dataset_Base, agent: BaseAgent, config: List[Dict[str, Any]]):
        """
        Initializes the agent. 
        The NCA does not not need to already have been moved to the correct device
        """
        self.dataset = dataset
        self.dataset_wrapper = DatasetWrapperVis(self.dataset)
        self.agent = agent
        self.config = config
        device = torch.device(config[0]['device'])
        self.device = device
        self.agent.model = self.agent.model.to(device)
        exp = Experiment_vis(self.config, self.dataset, self.agent.model, self.agent)
        self.dataset.set_experiment(experiment=exp)
        exp.set_model_state('train')

    def get_source_image_for_id(self, id: int) -> np.ndarray:
        """
        Returns fresh 3D image for ID (same as given by data loader)
        Image is already a normalized 3D np array and mostly ready for display.
        """
        _, fresh_image, _ = self.dataset.__getitem__(id)
        rescale = torchio.RescaleIntensity(out_min_max=(0,1), percentiles=(0.5, 99.5))
        if len(fresh_image.shape) == 4:
            fresh_image = fresh_image[..., 0]
        fresh_image = np.expand_dims(fresh_image, axis=0)
        fresh_image = rescale(fresh_image) 
        fresh_image = np.squeeze(fresh_image)
        return fresh_image
    
    def get_source_image_for_name(self, image: str) -> np.ndarray:
        """
        Returns image under given name as ndarray. 

        Only returns input Image, not mask or ID. 
        Image gets preprocessed for display beforehand.
        
        Returns: 
        Image as ndarray, without extra dimensions
        """
        image: np.ndarray
        with self.dataset_wrapper.image_in_dataset(image):
            image = self.get_source_image_for_id(0)
        return image

        
    def render_slice(self, src_image: np.ndarray, prediction: torch.Tensor, label: torch.Tensor, storepath: str):
        """
        Renders Slices for source image, prediction and label into an image and saves it to disc.

        Attributes:
        src_image: 2D slice of sourc image (input data)
        prediction: 2D slice of Network prediction
        label: 2D slice of corresponding label
        storepath: fully qualified storage path (including file name)
        """
        image = merge_img_label_gt(src_image, torch.sigmoid(prediction).numpy(), label.numpy())
        image = Image.fromarray(np.uint8(np.squeeze(image)*255)).convert('RGB')
        image.save(storepath, "PNG")
        
 

    def get_output_for_image_monitored(self, image: str, output_path: str, instrumentation_function: Callable[[np.ndarray, int], bool] = None, altered_input: np.ndarray = None, 
                                       save_images: bool = True) -> Dict[int, np.ndarray]:
        if not isinstance(self.agent.model, BasicNCA3DVis):
            raise Exception("Instrumentation is only possible while using an augmented model class")
        model: BasicNCA3DVis = self.agent.model
        model.set_instrumentation_function(instrumentation_function)
        model.set_state_dict({})
        self.get_output_for_image(image, output_path, altered_input, save_image=save_images)
        return model.export_state_dict()

    def get_output_for_image(self, image: str, output_path: str, altered_input: np.ndarray = None, save_image: bool = True):
        """
        Renders all slices for prediction for specified image.

        Paramaters:
        image: file name of image from dataset
        output_path: path to folder in which outputs are to be stored
        altered_input: Optional altered input image for whcih output is computed instead 
        of the correspnding image from the dataset. Intended to manually alter the dataset. 
        The image has to be already preprocessed. the get_source_image methods handle all the preprocessing necessary. 
        When altering values, final intensities have to be in [0,1]
        """
        with torch.no_grad():
            with self.dataset_wrapper.image_in_dataset(image):
                data = self.dataset.__getitem__(0)
                fresh_image = self.get_source_image_for_id(0)
                id, input, label = data
                if altered_input is not None:
                    if altered_input.shape != input.shape:
                        raise Exception("Shape of altered input image does not match shape of original image")
                    input = altered_input
                    fresh_image = copy.deepcopy(input)
                input = np.expand_dims(input, axis=0)
                label = np.expand_dims(label, axis=0)
                input = torch.from_numpy(input).to(self.device)
                label = torch.from_numpy(label).to(self.device)
                data = (id, input, label)
                _, inputs, _ = data
                data = self.agent.prepare_data(data)   

                outputs, targets = self.agent.get_outputs(data)
                patient_3d_image = outputs.detach().cpu()
                patient_3d_label = targets.detach().cpu()
                if save_image:
                    for m in range(patient_3d_image.shape[-1]):
                        if len(patient_3d_label.shape) == 4:
                            patient_3d_label = patient_3d_label.unsqueeze(dim=-1)
                        for i in range(0, fresh_image.shape[2]):
                            self.render_slice(fresh_image[:,:,i:i+1],patient_3d_image[:,:,:,i:i+1,m] ,patient_3d_label[:,:,:,i:i+1,m] , join(output_path, "out_" + str(i) + ".png"))
                    
            
    
    def dummy_output2(self, output_path: str):
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                """if oo < 4:
                    oo += 1
                    continue"""
                name, inputs, _ = data
                print("EWYOYOYOOYOYOYOYOYOOOOOOOOOOOOOOOOOOOO")
                print(name)
                data = self.agent.prepare_data(data)

                fresh_image = self.get_source_image_for_id(i)
                
                

                outputs, targets = self.agent.get_outputs(data)
                patient_3d_image = outputs.detach().cpu()
                patient_3d_label = targets.detach().cpu()
                patient_3d_real_Img = inputs.detach().cpu()
                for m in range(patient_3d_image.shape[-1]):
                    # Add image to tensorboard
                    if True: 
                        if len(patient_3d_label.shape) == 4:
                            patient_3d_label = patient_3d_label.unsqueeze(dim=-1)
                        middle_slice = int(patient_3d_real_Img.shape[3] /2)
                        
                        #image = merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy())
                        """ for i in range(0, out_np.shape[2]):
                        in_slice = im0.take(indices=i, axis=2)
                        out_slice = out_np.take(indices=i, axis=2)
                        target_slice = target_np.take(indices=i, axis=2)
                        # vis._write_im_2(in_slice, out_slice, target_slice, join(r"/home/nihm/NCA-3d/NCA/aaa", "out_" + str(i) + ".png"))"""
                        for i in range(0, patient_3d_real_Img.shape[3]):
                            #image = merge_img_label_gt(patient_3d_real_Img[:,:,:,i:i+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,i:i+1,m]).numpy(), patient_3d_label[:,:,:,i:i+1,m].numpy())
                            self.render_slice(fresh_image[:,:,i:i+1],patient_3d_image[:,:,:,i:i+1,m] ,patient_3d_label[:,:,:,i:i+1,m] , join(output_path, "out_" + str(i) + ".png"))
                           
                break






    def dummy_output(self, output_path: str):
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                """if oo < 4:
                    oo += 1
                    continue"""
                _, inputs, _ = data
                data = self.agent.prepare_data(data)

                _, fresh_image, _ = self.dataset.__getitem__(i)
                rescale = torchio.RescaleIntensity(out_min_max=(0,1), percentiles=(0.5, 99.5))
                if len(fresh_image.shape) == 4:
                    fresh_image = fresh_image[..., 0]
                fresh_image = np.expand_dims(fresh_image, axis=0)
                fresh_image = rescale(fresh_image) 
                fresh_image = np.squeeze(fresh_image)
                
                

                outputs, targets = self.agent.get_outputs(data)
                patient_3d_image = outputs.detach().cpu()
                patient_3d_label = targets.detach().cpu()
                patient_3d_real_Img = inputs.detach().cpu()
                for m in range(patient_3d_image.shape[-1]):
                    # Add image to tensorboard
                    if True: 
                        if len(patient_3d_label.shape) == 4:
                            patient_3d_label = patient_3d_label.unsqueeze(dim=-1)
                        middle_slice = int(patient_3d_real_Img.shape[3] /2)
                        
                        #image = merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy())
                        """ for i in range(0, out_np.shape[2]):
                        in_slice = im0.take(indices=i, axis=2)
                        out_slice = out_np.take(indices=i, axis=2)
                        target_slice = target_np.take(indices=i, axis=2)
                        # vis._write_im_2(in_slice, out_slice, target_slice, join(r"/home/nihm/NCA-3d/NCA/aaa", "out_" + str(i) + ".png"))"""
                        for i in range(0, patient_3d_real_Img.shape[3]):
                            #image = merge_img_label_gt(patient_3d_real_Img[:,:,:,i:i+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,i:i+1,m]).numpy(), patient_3d_label[:,:,:,i:i+1,m].numpy())
                            image = merge_img_label_gt(fresh_image[:,:,i:i+1], torch.sigmoid(patient_3d_image[:,:,:,i:i+1,m]).numpy(), patient_3d_label[:,:,:,i:i+1,m].numpy())
                            arr1 = patient_3d_real_Img[:,:,:,i:i+1].numpy()
                            arr2 = fresh_image[:,:,i:i+1]
                            image = Image.fromarray(np.uint8(np.squeeze(image)*255)).convert('RGB')
                            image.save(join(output_path, "out_" + str(i) + ".png"), "PNG")
                break





