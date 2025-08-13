import tifffile
from torch.utils.data import Dataset
from src.datasets.Data_Instance import Data_Container
from src.datasets.Dataset_Base import Dataset_Base
import cv2
import numpy as np
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Resize
from PIL import Image
from os import listdir
from os.path import join
import os
import random, zarr, tqdm

class Dataset_BCSS_Seg(Dataset_Base):
    def __init__(self, fixed_patches: bool, patch_size: tuple[int, int], images_path) -> None:
        self.slice = -1
        self.delivers_channel_axis = True
        self.is_rgb = True
        self.fixed_patches = fixed_patches
        if self.fixed_patches:
            self.patches = {}
            for f in tqdm.tqdm(os.listdir(images_path)):
                self.patches[f] = []
                width, height, _ = zarr.open(os.path.join(images_path, f)).shape
                for x in range(0, width-patch_size[0], patch_size[0]):
                    for y in range(0, height-patch_size[1], patch_size[1]):
                        self.patches[f].append((x, y))



    def getFilesInPath(self, path: str):
        files = os.listdir(path)
        dic = {}
        for f in files:
            dic[f]={}
            if self.fixed_patches:
                for x,y in self.patches[f]:
                    dic[f][f"{x}_{y}"] = (f,x,y)
            else:
                dic[f][0] = f
        return dic
    
    def __getitem__(self, idx: str):
        if self.fixed_patches:
            f,x,y = self.images_list[idx]
            _id = f"{f}_{x}_{y}"
            mmapped_image = zarr.open(os.path.join(self.images_path, f))
            mmapped_label = zarr.open(os.path.join(self.labels_path, f))
            pos_x = x
            pos_y = y
        else:
            _id = self.images_list[idx]
            file_path = os.path.join(self.images_path, self.images_list[idx])
            label_path = os.path.join(self.labels_path, self.labels_list[idx])
            mmapped_image = zarr.open(file_path, mode='r')
            mmapped_label = zarr.open(label_path, mode='r')
            pos_x = random.randint(0, mmapped_image.shape[0] - self.size[0])
            pos_y = random.randint(0, mmapped_image.shape[1] - self.size[1])

        assert mmapped_image.shape[:2] == mmapped_label.shape[:2], f"{mmapped_image.shape[:2]} != {mmapped_label.shape[:2]}"


        img = mmapped_image[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1]]
        lbl = mmapped_label[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1]]

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img / 255.0
        img = (img - mean) / std

        if len(lbl.shape) == 2:
            lbl = lbl[..., None]

        data_dict = {}
        data_dict['id'] = _id
        data_dict['image'] = img
        data_dict['label'] = lbl
        #2D: data_dict['image']: HWC, data_dict['label']: HWC

        return data_dict