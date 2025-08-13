import PIL
import openslide.deepzoom
import tifffile
from torch.utils.data import Dataset
import tqdm
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
import random

import zarr
import openslide
import pickle as pkl

class Dataset_AGGC(Dataset_Base):
    #https://aggc22.grand-challenge.org/Data/
    def __init__(self, patches_path, label_path, img_path, patch_size, merge_all_classes: bool) -> None:
        super().__init__()
        self.slice = 2
        self.delivers_channel_axis = True
        self.is_rgb = True
        self.merge_all_classes = merge_all_classes

        self.patches_path = patches_path
        if os.path.exists(os.path.join(self.patches_path, f"patches_{patch_size[0]}_{patch_size[1]}.pkl")):
            print("loading patches")
            self.patches = pkl.load(open(os.path.join(self.patches_path, f"patches_{patch_size[0]}_{patch_size[1]}.pkl"), "rb"))
        else:
            print("patchifying")
            self.patches = {}
            num_skipped_patches = {f: 0 for f in os.listdir(img_path)}

            num_for_loop_items = 0
            for f in tqdm.tqdm(os.listdir(img_path)):
                width, height = openslide.OpenSlide(os.path.join(img_path, f)).level_dimensions[0]
                for x in range(0, width-patch_size[0], patch_size[0]):
                    for y in range(0, height-patch_size[1], patch_size[1]):
                        num_for_loop_items += 1



            bar = tqdm.tqdm(range(num_for_loop_items))
            bar_iter = iter(bar)
            for f in os.listdir(img_path):
                self.patches[f] = []
                width, height = openslide.OpenSlide(os.path.join(img_path, f)).level_dimensions[0]
                mmapped_lbls = []
                for i, seg_class in enumerate(["Stroma", "Normal", "G3", "G4", "G5"]):
                    label_path = os.path.join(label_path, f[:-len(".tiff")], f"{seg_class}_Mask.zarr")
                    if not os.path.exists(label_path):
                        mmapped_lbls.append(None)
                        continue
                    mmapped_lbl = zarr.open(label_path, mode='r')
                    assert mmapped_lbl.shape == (width, height), f"{mmapped_lbl.shape} != {(width, height)}"
                    mmapped_lbls.append(mmapped_lbl)

                for x in range(0, width-patch_size[0], patch_size[0]):
                    bar.set_description(f"{f} x={x}")
                    for y in range(0, height-patch_size[1], patch_size[1]):
                        next(bar_iter)
                        label = np.zeros((patch_size[0], patch_size[1], 5), dtype=int)
                        for i, mmapped_lbl in enumerate(mmapped_lbls):
                            if mmapped_lbl is None:
                                continue
                            mmapped_lbl = mmapped_lbl[x:x+patch_size[0], y:y+patch_size[1]]
                            label[:,:,i] = mmapped_lbl

                        label_sum = np.sum(label, axis=2)
                        #print(label.sum(), np.count_nonzero(label_sum))
                        if np.count_nonzero(label_sum) / (patch_size[0] * patch_size[1]) < 0.1: # 10% of the patch must be labeled
                            num_skipped_patches[f] += 1
                            continue
                        else:
                            self.patches[f].append((x, y))
                            pass
            bar.close()
            os.makedirs(self.patches_path, exist_ok=True)
            pkl.dump(self.patches, open(os.path.join(self.patches_path, f"patches_{patch_size[0]}_{patch_size[1]}.pkl"), "wb"))
            print(num_skipped_patches)
            print({k: len(v) for k,v in self.patches.items()})
            exit()



    def getFilesInPath(self, path: str):
        files = os.listdir(path)
        dic = {}
        for f in files:
            if not f.endswith(".tiff"):
                f = f + ".tiff"
            if len(self.patches[f]) == 0:
                continue
            dic[f]={}
            for x,y in self.patches[f]:
                dic[f][f"{x}_{y}"] = (f,x,y)
        return dic

    def __getitem__(self, idx: str):
        file_name, x, y = self.images_list[idx]
        file_path = os.path.join(self.images_path, file_name)
        openslide_image = openslide.OpenSlide(file_path)

        #openslide_image.level_dimensions = (W, H)
        width, height = openslide_image.level_dimensions[0]

        pos_x = x
        pos_y = y

        img = openslide_image.read_region((pos_x, pos_y), 0, self.size)

        label = np.zeros((self.size[0], self.size[1], 5), dtype=int)

        for i, seg_class in enumerate(["Stroma", "Normal", "G3", "G4", "G5"]):
            label_path = os.path.join(self.labels_path, file_name[:-len(".tiff")], f"{seg_class}_Mask.zarr")
            if not os.path.exists(label_path):
                continue

            mmapped_lbl = zarr.open(label_path, mode='r')
            #print(file_path, label_path)
            assert mmapped_lbl.shape[0] == width, f"{mmapped_lbl.shape[0]} != {openslide_image.level_dimensions[0][0]}"
            assert mmapped_lbl.shape[1] == height, f"{mmapped_lbl.shape[1]} != {openslide_image.level_dimensions[0][1]}"
            #print(mmapped_lbl.shape, openslide_image.level_dimensions[0])
            #exit()

            mmapped_lbl = mmapped_lbl[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1]]

            label[:,:,i] = mmapped_lbl

        if self.merge_all_classes:
            label = np.sum(label, axis=2)
            label = label > 0
            label = label[:, :, None]

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = np.array(img, dtype=float)
        img = img[:,:,:3]
        img = img / 255.0
        img = (img - mean) / std

        data_dict = {}
        data_dict['id'] = f"{file_name}_{x}_{y}"
        data_dict['image'] = img
        data_dict['label'] = label


        #2D: data_dict['image']: HWC, data_dict['label']: HWC

        return data_dict