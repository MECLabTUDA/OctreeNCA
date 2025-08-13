

import einops
from src.datasets.Dataset_3D import Dataset_3D
from src.datasets.Dataset_Base import Dataset_Base
import os
import numpy as np
import cv2
import torch, math

class Dataset_CholecSeg_preprocessed(Dataset_Base):
    def __init__(self, patch_size: tuple=None, slice_wise: bool = False):
        super().__init__()
        self.slice = None
        self.delivers_channel_axis = True
        self.is_rgb = True

        self.patch_size = patch_size
        self.slice_wise = slice_wise
        if self.slice_wise:
            self.slice = 2

    
    def getFilesInPath(self, path: str):
        r"""Get files in path
            Args:
                path (string): The path which should be worked through
            Returns:
                dic (dictionary): {key: patient_name, value: {key: index, value: file_name}}
        """
        dirs = os.listdir(path)
        dic = {}
        for inner_dir in dirs:
            if not os.path.isdir(os.path.join(path, inner_dir)):
                continue
            id = inner_dir
            dic[id] = {}
            for i, f in enumerate(os.listdir(os.path.join(path, inner_dir))):
                if self.slice_wise:
                    for j in range(80):
                        dic[id][i*80+j] = (f, j)
                else:
                    dic[id][i] = f
        return dic
    


    def load_item_internal(self, path):
        imgs = np.load(os.path.join(path, "video.npy"))#CHWD
        lbls = np.load(os.path.join(path, "segmentation.npy"))#HWDC

        def reshape_batch(instack, is_label:bool = False) -> np.ndarray:
            #https://stackoverflow.com/questions/65154879/using-opencv-resize-multiple-the-same-size-of-images-at-once-in-python
            N,H,W,C = instack.shape
            instack = instack.transpose((1,2,3,0)).reshape((H,W,C*N))

            outstacks = []
            for i in range(math.ceil(instack.shape[-1] / 500)):
                if is_label:
                    outstack = cv2.resize(instack[..., i*500:(i+1)*500], (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    outstack = cv2.resize(instack[..., i*500:(i+1)*500], (self.size[1], self.size[0]))
                outstacks.append(outstack)

            outstack = np.concatenate(outstacks, axis=-1)
            return outstack.reshape((self.size[0], self.size[1], C, N)).transpose((3,0,1,2))


        imgs = reshape_batch(imgs.transpose(3,1,2,0))
        lbls = reshape_batch(lbls.transpose(2,0,1,3), is_label=False)

        imgs = imgs.transpose(3, 1, 2, 0)#DHWC -> CHWD
        lbls = lbls.transpose(1, 2, 0, 3)#DHWC -> HWDC

        data_dict = {}
        data_dict['image'] = imgs
        data_dict['label'] = lbls
        return data_dict

    def setState(self, state: str) -> None:
        super().setState(state)

    def __getitem__(self, idx: str):
        # images have a resolution of 854x480 with 80 frames

        if self.slice_wise:
            file_name, slice_idx = self.images_list[idx]
            patient_name = file_name[:len("videoXX")]
        else:
            file_name = self.images_list[idx]
            patient_name = file_name[:len("videoXX")]

        path = os.path.join(self.images_path, patient_name, file_name)

        data_dict = self.load_item_internal(path)
        data_dict['patient_id'] = patient_name
        data_dict['recording_id'] = file_name


        data_dict['id'] = file_name

        # patient_id: id that indicates the patient (used for data splitting)
        # recording_id: id that indicates the recording (this is already unique for all patients) 
        #       the recording id used during evaluation. Each recording id needs to be predicted separately
        # id: some id, that is unique for each sample

        if self.slice_wise:
            assert self.patch_size is None, "Patch size is not supported for slice wise"
            data_dict['id'] = f"{file_name}_{slice_idx}"
            data_dict['image'] = data_dict['image'][...,slice_idx]
            data_dict['label'] = data_dict['label'][:,:,slice_idx]

            data_dict['image'] = einops.rearrange(data_dict['image'], 'c h w -> h w c')
            data_dict['label'] = einops.rearrange(data_dict['label'], 'h w c -> h w c')
        
        if self.patch_size is not None and self.state == "train":
            img = data_dict['image']
            lbl = data_dict['label']
            if img.shape[1] == self.patch_size[0]:
                x = 0
            else:
                x = np.random.randint(0, img.shape[1] - self.patch_size[0])
            
            if img.shape[2] == self.patch_size[1]:
                y = 0
            else:
                y = np.random.randint(0, img.shape[2] - self.patch_size[1])

            if img.shape[3] == self.patch_size[2]:
                z = 0
            else:
                z = np.random.randint(0, img.shape[3] - self.patch_size[2])
            
            img = img[:, x:x+self.patch_size[0], y:y+self.patch_size[1], z:z+self.patch_size[2]]
            lbl = lbl[x:x+self.patch_size[0], y:y+self.patch_size[1], z:z+self.patch_size[2]] 
            data_dict['image'] = img
            data_dict['label'] = lbl


        return data_dict
    
    def setPaths(self, images_path: str, images_list: str, labels_path: str, labels_list: str) -> None:
        return super().setPaths(images_path, images_list, labels_path, labels_list)
    
    def set_size(self, size: tuple) -> None:
        super().set_size(size)
        assert self.size[2] == 80, f"The temporal dimension must be 80, but got {self.size}"