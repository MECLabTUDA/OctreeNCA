import einops
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
import random, zarr, tqdm, pickle as pkl, openslide

class Dataset_PESO(Dataset_Base):
    def __init__(self, patches_path: str, patch_size: tuple[int, int], path: str, img_level:int, 
                 return_background_class: bool=False) -> None:
        self.slice = -1
        self.delivers_channel_axis = True
        self.is_rgb = True

        self.img_level = img_level
        self.return_background_class = return_background_class

        assert len(patch_size) == 2, f"patch_size must be a tuple of length 2, but is {patch_size}"
        

        if os.path.exists(os.path.join(patches_path, f"patches_{img_level}_{patch_size[0]}_{patch_size[1]}.pkl")):
            print("loading patches")
            self.patches = pkl.load(open(os.path.join(patches_path, f"patches_{img_level}_{patch_size[0]}_{patch_size[1]}.pkl"), "rb"))
        else:
            print("patchifying")
            self.patches = {}
            segmentation_file_names = list(filter(lambda x: x.endswith("_training_mask.tif"), os.listdir(path)))
            num_skipped_patches = {f[:-len("_training_mask.tif")]: 0 for f in segmentation_file_names}

            num_for_loop_items = 0
            for f in tqdm.tqdm(segmentation_file_names):
                image_lbl = openslide.OpenSlide(os.path.join(path, f))
                assert img_level < len(image_lbl.level_dimensions), f"img_level {img_level} >= len(image_lbl.level_dimensions) {len(image_lbl.level_dimensions)}"
                width, height = image_lbl.level_dimensions[img_level]
                for x in range(0, width-patch_size[0], patch_size[0]):
                    for y in range(0, height-patch_size[1], patch_size[1]):
                        num_for_loop_items += 1


            bar = tqdm.tqdm(range(num_for_loop_items))
            bar_iter = iter(bar)
            for f in segmentation_file_names:
                name = f[:-len("_training_mask.tif")]
                self.patches[name] = []
                image_lbl = openslide.OpenSlide(os.path.join(path, f))
                width, height = image_lbl.level_dimensions[img_level]

                for x in range(0, width-patch_size[0], patch_size[0]):
                    bar.set_description(f"{name} x={x}")
                    for y in range(0, height-patch_size[1], patch_size[1]):
                        next(bar_iter)

                        slide_seg = openslide.open_slide(os.path.join(path, f))
                        seg = slide_seg.read_region((int(x * slide_seg.level_downsamples[img_level]), 
                                                     int(y * slide_seg.level_downsamples[img_level])), img_level, patch_size)
                        seg = np.array(seg)[:,:,0] #channel RGB contains the same values, only alpha might be different
                        assert seg.shape == tuple(patch_size), f"{seg.shape} != {patch_size}"
                        
                        #if np.count_nonzero(seg == 2) / (patch_size[0] * patch_size[1]) < 0.01: # 1% of the patch must be labeled
                        if np.count_nonzero(seg == 2) == 0: # 1% of the patch must be labeled
                            num_skipped_patches[name] += 1
                            continue
                        else:
                            self.patches[name].append((x, y))
                            pass
            bar.close()
            os.makedirs(patches_path, exist_ok=True)
            pkl.dump(self.patches, open(os.path.join(patches_path, f"patches_{img_level}_{patch_size[0]}_{patch_size[1]}.pkl"), "wb"))
            print("skipped patches:")
            print(num_skipped_patches)
            print("patches:")
            print({k: len(v) for k,v in self.patches.items()})
            exit()


    def getFilesInPath(self, path: str):
        files = filter(lambda x: x.endswith("_training_mask.tif"), os.listdir(path))
        dic = {}
        for f in files:
            name = f[:-len("_training_mask.tif")]
            dic[name]={}
            for x,y in self.patches[name]:
                dic[name][f"{x}_{y}"] = (name,x,y)
        return dic
    
    def __getitem__(self, idx: str):
        name,x,y = self.images_list[idx]
        _id = f"{name}_{x}_{y}"
        img_slide = openslide.open_slide(os.path.join(self.images_path, f"{name}.tif"))
        lbl_slide = openslide.open_slide(os.path.join(self.labels_path, f"{name}_training_mask.tif"))
        pos_x = x
        pos_y = y

        img = img_slide.read_region((int(pos_x * img_slide.level_downsamples[self.img_level]),
                                        int(pos_y * img_slide.level_downsamples[self.img_level])), 
                                        self.img_level, self.size)

        lbl = lbl_slide.read_region((int(pos_x * lbl_slide.level_downsamples[self.img_level]),
                                        int(pos_y * lbl_slide.level_downsamples[self.img_level])), 
                                        self.img_level, self.size)

        img = np.array(img)[:,:,0:3]
        lbl = np.array(lbl)[:,:,0]

        lbl = lbl == 2
        lbl = lbl



        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img / 255.0
        img = (img - mean) / std


        assert len(lbl.shape) == 2, f"lbl.shape must be H W, but is {lbl.shape}"
        if self.return_background_class:
            background_lbl = np.logical_not(lbl)
            lbl = np.stack([background_lbl, lbl], 2)
        else:
            lbl = einops.rearrange(lbl, "h w -> h w 1")

        lbl = lbl.astype(float)
        #print(lbl.shape)   #HWC


        data_dict = {}
        data_dict['id'] = _id
        data_dict['image'] = img
        data_dict['label'] = lbl
        data_dict['patient_id'] = name
        data_dict['position'] = (pos_x, pos_y)
        #2D: data_dict['image']: HWC, data_dict['label']: HWC

        return data_dict
    
    def getPublicIdByIndex(self, idx: int):
        name,x,y = self.images_list[idx]
        _id = f"{name}_{x}_{y}"
        return _id