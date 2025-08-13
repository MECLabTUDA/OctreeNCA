from src.datasets.Dataset_Base import Dataset_Base
from src.datasets.Data_Instance import Data_Container
from src.utils.Experiment_vis import Gutted_Experiment
from src.utils.DatasetClassCreator import Extended_Loadable_Dataset_Annotations
from typing import Dict, Any, Tuple
from contextlib import contextmanager
import os
"""
DEPRECATED
DEPRECATED
DEPRECATED
DO NOT USE
DO NOT USE
DO NOT USE

"""


class DatasetWrapperVis():
    """
    Wrapper class for Dataset for visualization. 
    Data is always loaded from disk. 
    This igores data splits. 
    Allows loading by name.
    ONLY TEMPORARY, very scuffed there probably is a better way to do this but it does not seem to break anything
    """
    dataset: Dataset_Base
    def __init__(self, dataset: Dataset_Base):
        self.dataset = dataset

    @contextmanager
    def image_in_dataset(self, filename: str):
        temp_labels = self.dataset.labels_list
        temp_images = self.dataset.images_list
        temp_container = self.dataset.data
        try:

            self.dataset.data = Data_Container()
            self.dataset.labels_list = {0: (filename, 0, 0)} 
            self.dataset.images_list = {0: (filename, 0, 0)}
            # entries in the dataset are of the form: 
            #(image_name, patient_id, image_id)
            # image_name is just a file name
            # patient_id has to be unqique for each image in the dataset
            # image_id addresses individual slices of the image. If 3D input desired always 0
            
            yield (0, filename)
        finally:
            self.dataset.data = temp_container
            self.dataset.images_list = temp_images
            self.dataset.labels_list = temp_labels
            
    @classmethod
    def add_required_to_config(clas, config: Dict[str, Any]):
        r"""Fills config with basic setup if not defined otherwise
        @TODO find a better place for this
        """
        if 'Persistence' not in config:
            config['Persistence'] = False
        if 'batch_duplication' not in config:
            config['batch_duplication'] = 1
        if 'keep_original_scale' not in config:
            config['keep_original_scale'] = False
        if 'rescale' not in config:
            config['rescale'] = True
        if 'channel_n' not in config:
            config['channel_n'] = 16
        if 'cell_fire_rate' not in config:
            config['cell_fire_rate'] = 0.5
        if 'output_channels' not in config:
            config['output_channels'] = 1
            
    @classmethod
    def set_arbitrary_dataset(cls, dataset: Dataset_Base, image_path: str, labels_path: str, config: Dict[str, Any], slices: int, slice_axis: int = None)-> Tuple[Dataset_Base, Dict[str, Any]]:
        """Takes a Dataset Object and sets an arbitrary underlying dataset for it

        Args:
            image_path (str): Fully qualified path to Folder containing source images
            labels_path (str): Fully qualified path to Folder containing labels corresponding to source images. 
            Both folder have to contain only images from the same Dataset. Corresponding labels and images need equal names.
            slices (int): Sets whether the underlying 3D images should be sliced, and gives length of the axis (e.g. how many slices per Image).
            config: config
        Returns:
            Returns Dict
            maps file_name -> list(unique_ids). List of unique id's using which the individual slices of the source 
            image can be quarried with __get__item. id's correspond to individual slices of the source image in ascending order. 
            
        """
        DatasetWrapperVis.add_required_to_config(config)
        f_names = os.listdir(image_path)
        f_names_labels = os.listdir(labels_path)
        if len(f_names) != len(f_names_labels):
            raise Exception("Number of labels and images has to match")
        retDict: Dict[str, Any] = {}
        dataset.data = Data_Container()

        dataset.slice = slice_axis # if the dataset didn't have a slice attribute before it has now. 
        # This is not nice, but python allows it
        dataset.exp = Gutted_Experiment(config=config)
        retDict: Dict[str, Any] = {}
        labels_dict = {}
        images_dict = {}
        unique_id: int = -1
        for f_name in f_names:
            image_slice_list = list()
            patient_id = unique_id + 1
            if slice_axis is None:
                unique_id += 1
                image_slice_list.append(unique_id)
                labels_dict[unique_id] = (f_name, unique_id, 0)
                images_dict[unique_id] = (f_name, unique_id, 0)
            
            else:
                for i in range(0, slices):
                    unique_id += 1
                    image_slice_list.append(unique_id)
                    labels_dict[unique_id] = (f_name, patient_id, i)
                    images_dict[unique_id] = (f_name, patient_id, i)
            retDict[f_name] = image_slice_list
        dataset.setPaths(image_path, images_dict, labels_path, labels_dict)
        dataset.size = config['input_size']
        return (dataset, retDict)
            
            
                    
        