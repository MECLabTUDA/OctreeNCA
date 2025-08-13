from typing import Type, Dict, Any, TYPE_CHECKING, List, Union, TypeVar, Optional, Tuple
import torchio
import numpy as np
from src.datasets.Dataset_Base import Dataset_Base
import types
from src.datasets.Data_Instance import Data_Container
import os
"""
This file contains infrastructure that supports the dynamic 
and (relatively) painless creation of new dataset classes that allow 
for easy loading of arbitrary Datasets without using experiments. 
Type hints work with pylance, untested using other development tools.
    

Datasets should be used together with Visualization agents and models.
"""
class Extended_Loadable_Dataset_Annotations(Dataset_Base):
    if TYPE_CHECKING:
        def get_from_config(self, tag: str) -> Any:
                r"""Get from config
                    #Args
                        tag (String): Key of requested value
                """
                pass
        def get_dataset_index_information(self) -> Dict[str, List[int]]:
            """
            Returns:
                    Returns Dict
                    maps file_name -> list(unique_ids). List of unique id's using which the individual slices of the source 
                    image can be quarried with __get__item. id's correspond to individual slices of the source image in ascending order. 
                    
            """
            pass
        def __init__(self, image_path: str, labels_path: str, config: Dict[str, Any], slices: int = None, slice_axis: int = None, is_3d: bool = True):
            """Takes a Dataset Object and sets an arbitrary underlying dataset for it

            Args:
                image_path (str): Fully qualified path to Folder containing source images
                labels_path (str): Fully qualified path to Folder containing labels corresponding to source images. 
                Both folder have to contain only images from the same Dataset. Corresponding labels and images need equal names.
                slices (int): Sets whether the underlying 3D images be sliced along an axis for display 
                purposes, and gives length of the axis (e.g. how many slices per Image).
                config: config
                is_3D: whether the network receives 3D data or individual slices along the axis
            
            """
            pass
        
        def get_ids_for_filename(self, fname: str) -> list(int):
            """
            Returns all retreival IDs that correspond to given filename
            """ 
            pass


        def get_dataset_index_for_filename_slice(self, fname: str, slice_num: int = 0) -> int:
            """Returns the index that can be used with __get_item__ to retreive the specified slice of the 
            corresponding filename. 
            Also works for 3D, non-sliced images. the slice parameter is ignored in this case."""
        
        def get_source_image_for_id(self, id: int) -> np.ndarray:
            """
            Returns fresh 3D image for ID (same as given by data loader)
            Image is already a normalized 3D np array and mostly ready for display.
            """
            pass
        
        def get_state(self) -> Tuple[Any]:
            """I do not like this. This returns the things that I assume are important for the internal state? 
            Who knows whether I'm right..

            Returns:
                Tuple[Any]: things i think might be important
            """
            
        def set_state(self, state: Tuple[Any]):
            """Tries to set the internal state. I again very much do not like this. 
            However I need to do this as something in the process of setting things changes 
            the data splits. I assume this might happen when the experiment tries to reset the datasplits?
            But I do not want to touch this function in the Experiment_Vis as this seems like a very good way to break things.

            Args:
                state (Tuple[Any]): _description_

            Raises:
                Exception: _description_

            Returns:
                _type_: _description_
            """

        def does_network_receive_3d(self) -> bool:
            """
            Returns whether the network wants to receive 3D data or individual slices.
            """


        def get_source_image_for_filename(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
            """
            Returns the whole source image for the given file name as well as the whole label. 
            Independant on whether the dataset produces sliced data
            """
        
        
        
class DatasetClassCreator():
    
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
    def create_loadable_dataset_class_from_class(cls, baseClass: Type[Dataset_Base]) -> Type[Extended_Loadable_Dataset_Annotations]:
        
        def anonymous_constructor(self, image_path: str, labels_path: str, config: Dict[str, Any], slices: int = None, slice_axis: int = None):
            """Takes a Dataset Object and sets an arbitrary underlying dataset for it

            Args:
                image_path (str): Fully qualified path to Folder containing source images
                labels_path (str): Fully qualified path to Folder containing labels corresponding to source images. 
                Both folder have to contain only images from the same Dataset. Corresponding labels and images need equal names.
                slices (int): Sets whether the underlying 3D images should be sliced, and gives length of the axis (e.g. how many slices per Image).
                DEPRECATED: Slice informaiton is now generated using the information provided in the base class.
                slice_axis: equivalent to slice parameter in config. If set to None: Agent receives 3d Data.
                config: config
            
            """
            super(NewDatasetClass, self).__init__()
            self.config = config
            DatasetClassCreator.add_required_to_config(config)
            f_names = os.listdir(image_path)
            f_names_labels = os.listdir(labels_path)
            if len(f_names) != len(f_names_labels):
                raise Exception("Number of labels and images has to match")
            retDict: Dict[str, Any] = {}
            self.data = Data_Container()
            self.size = config["input_size"]
            self.slice = slice_axis # if the dataset didn't have a slice attribute before it has now. 
            # This is not nice, but python allows it
            #self.exp = Gutted_Experiment(config=config)
            
            
            @classmethod
            def get_from_config(self, tag: str) -> Any:
                r"""Get from config
                    #Args
                        tag (String): Key of requested value
                """
                if tag in self.config.keys():
                    return self.config[tag]
                else:
                    return None
            self.exp = type("AnonymousConfigWrapper", (object, ), {"get_from_config": get_from_config, "config": config})
            """
            retDict: Dict[str, Any] = {}
            labels_list = list()
            images_list = list()
            unique_id: int = -1
            for f_name in f_names:
                image_slice_list = list()
                patient_id = unique_id + 1
                if slice_axis is None:
                    unique_id += 1
                    image_slice_list.append(unique_id)
                    labels_list.append((f_name, unique_id, 0))
                    images_list.append((f_name, unique_id, 0))
                
                else:
                    for i in range(0, slices):
                        unique_id += 1
                        image_slice_list.append(unique_id)
                        labels_list.append((f_name, patient_id, i))
                        images_list.append((f_name, patient_id, i))
                retDict[f_name] = image_slice_list"""
            # uses implementation provided by base dataset.
            labels_dict = self.getFilesInPath(image_path)
            
            images_dict = self.getFilesInPath(labels_path)
            l_keys = list(labels_dict.keys())
            i_keys = list(images_dict.keys())
            if len(l_keys) != len(i_keys):
                raise Exception("Cannot handle unequal number of lists and labels")
            images = list()
            labels = list()
            running_index = 0
            fname_id_dict: Dict[str, List[int]] = {}
            for i in range(len(l_keys)):
                new_labels = list(labels_dict[l_keys[i]].values())
                new_images = list(images_dict[i_keys[i]].values())
                images.extend(new_images)
                labels.extend(new_labels)
                unique_ids = [*range(running_index, running_index+len(new_labels))]
                fname_id_dict[l_keys[i]] = unique_ids
                running_index += len(new_labels)
            
            self.setPaths(image_path, images, labels_path, labels)
            
            if isinstance(config['input_size'][0], tuple):
                self.size = config['input_size'][-1]
            else:
                self.size = self.config['input_size']
            # dataset info is used to efficiently retrieve data points for a given file
            self.dataset_info =  fname_id_dict
            


        def get_dataset_index_for_filename_slice(self, fname: str, slice_num: int = 0):
            """
            Returns index that is to be used with __getitem__ to retrieve the corresponding point. 
            If the Network receives unsliced 3D data, use slice_num=None
            """
            if not fname in self.dataset_info:
                return None
            if self.slice is None:
                return self.dataset_info[fname][0]
            else:
                if not slice_num in self.dataset_info[fname]:
                    return self.dataset_info[fname][0]
                else:
                    return self.dataset_info[fname][slice_num]
        def get_dataset_index_information(self):
            """
            Returns:
                    Returns Dict
                    maps file_name -> list(unique_ids). List of unique id's using which the individual slices of the source 
                    image can be queried with __get__item. id's correspond to individual slices of the source image in ascending order. 
                    
            """
            return self.dataset_info
        
        def get_ids_for_filename(self, fname: str):
            d_set = getattr(self, "dataset_info")
            if fname in d_set:
                return d_set[fname]
            else:
                return None
            
        def set_experiment(self, experiment: Any):
            pass



        
        def does_network_receive_3d(self) -> bool:
            """
            Returns whether the network wants to receive 3D data or individual slices.
            """
            return self.slice is None
        
        def get_state(self) -> Tuple[Any]:
            return(self.images_path,
                    self.images_list,
                    self.labels_path,
                    self.labels_list,
                    self.length,
                    self.size, 
                    self.dataset_info)
            
        def set_state(self, state: Tuple[Any]):
            images_path, images_list, labels_path, labels_list, length, size, dataset_info = state
            self.images_path = images_path
            self.images_list = images_list
            self.labels_path = labels_path
            self.labels_list = labels_list
            self.length = length
            self.size = size
            self.dataset_info = dataset_info
            
        def get_source_image_for_id(self, id: int) -> np.ndarray:
            """
            Returns fresh 3D image for ID (same as given by data loader)
            Image is already a normalized 3D np array and mostly ready for display.

            DEPRECATED maybe?? Probably best to not use this.
            """
            _, fresh_image, _ = self.__getitem__(id)
            rescale = torchio.RescaleIntensity(out_min_max=(0,1), percentiles=(0.5, 99.5))
            if len(fresh_image.shape) == 4:
                fresh_image = fresh_image[..., 0]
            fresh_image = np.expand_dims(fresh_image, axis=0)
            fresh_image = rescale(fresh_image) 
            fresh_image = np.squeeze(fresh_image)
            return fresh_image
        


        def get_source_image_for_filename(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
            """
            Returns the whole source image for the given file name as well as the whole label. 
            Independant of whether the dataset represents sliced data. This method should be used when retreiving whole Data Points. 
            """
            label = None
            if self.slice is None:
                ids = self.dataset_info[filename]
                _, fresh_image, ldummy = self.__getitem__(ids[0])
                
                label = ldummy
            else: 
                ids = self.dataset_info[filename]
                _, sdummy, ldummy = self.__getitem__(ids[0])
                source_dims_list = list(sdummy.shape)
                source_dims_list.insert(self.slice, len(ids))
                sfindims = tuple(source_dims_list)
                sretarr = np.zeros(sfindims)

                ldims_list = list(ldummy.shape)
                ldims_list.insert(self.slice, len(ids))
                lfindims = tuple(ldims_list)
                lretarr = np.zeros(lfindims)
                for i in range(len(ids)):
                    _, fresh_image, lbl = self.__getitem__(ids[i])
                    if self.slice == 0:
                        sretarr[i, ...] = fresh_image
                        lretarr[i, ...] = lbl
                    elif self.slice == 1:
                        sretarr[:, i, ...] = fresh_image
                        lretarr[:, i, ...] = lbl
                    elif self.slice == 2:
                        sretarr[:, :, i, ...] = fresh_image
                        lretarr[:, :, i, ...] = lbl
                label = lretarr
                fresh_image = sretarr
            rescale = torchio.RescaleIntensity(out_min_max=(0,1), percentiles=(0.5, 99.5))
            if len(fresh_image.shape) == 4:
                fresh_image = fresh_image[..., 0]
            fresh_image = np.expand_dims(fresh_image, axis=0)
            fresh_image = rescale(fresh_image) 
            fresh_image = np.squeeze(fresh_image)
            return (fresh_image, label)


                        

        

            

            
        NewDatasetClass = type("ConstructorInitializedDataset", (baseClass, ), {
            # constructor
            "__init__": anonymous_constructor, 
            
            # function members
            "get_dataset_index_information": get_dataset_index_information,
            "get_ids_for_filename": get_ids_for_filename,
            "get_source_image_for_id": get_source_image_for_id,
            "set_experiment": set_experiment,
            "get_state": get_state, 
            "set_state": set_state,
            "does_network_receive_3d": does_network_receive_3d,
            "get_dataset_index_for_filename_slice": get_dataset_index_for_filename_slice,
            "get_source_image_for_filename": get_source_image_for_filename,
            # object attribes
            "dataset_info": None,
            "config": None,
            "network_receive_3d": True
            
        })
        return NewDatasetClass
                
            
    