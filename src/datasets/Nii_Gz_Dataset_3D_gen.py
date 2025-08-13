from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
import nibabel as nib
import os
import numpy as np
import cv2
import random
import torchio
import matplotlib.pyplot as plt
import torch

class Dataset_NiiGz_3D_gen(Dataset_NiiGz_3D):
    """This dataset additionally stores an input vector that is updated"""

    def __init__(self, slice: int =None, resize: bool =True, store: bool = True, extra_channels = 8) -> None: 
        self.extra_channels = extra_channels
        super().__init__(slice, resize, store)

    def set_vec(self, idx: str, vec) -> tuple:
        #print("INDEX", idx)
        #print("VEC", vec)
        img = self.data.get_data(key=self.images_list[idx])
        id, img, label, img_vec = img
        img_vec = vec
        self.data.set_data(key=self.images_list[idx], data=(id, img, label, img_vec))



    def __getitem__(self, idx: str) -> tuple:
        r"""Standard get item function
            #Args
                idx (int): Id of item to loa
            #Returns:
                img (numpy): Image data
                label (numpy): Label data
        """
        rescale = torchio.RescaleIntensity(out_min_max=(0,1), percentiles=(0.5, 99.5))
        znormalisation = torchio.ZNormalization()
        #torch.manual_seed(idx)
        #random.seed(idx)
        #np.random.seed(idx)

        img = self.data.get_data(key=self.images_list[idx])
        if not img:
            img_name, p_id, img_id = self.images_list[idx]

            label_name, _, _ = self.labels_list[idx]

            img, label = self.load_item(os.path.join(self.images_path, img_name)), self.load_item(os.path.join(self.labels_path, img_name))
            # 2D
            if self.slice is not None:
                if len(img.shape) == 4:
                    img = img[..., 0]
                if self.exp.get_from_config('rescale') is not None and self.exp.get_from_config('rescale') is True:
                    img, label = self.rescale3d(img), self.rescale3d(label, isLabel=True)
                if self.slice == 0:
                    img, label = img[img_id, :, :], label[img_id, :, :]
                elif self.slice == 1:
                    img, label = img[:, img_id, :], label[:, img_id, :]
                elif self.slice == 2:
                    img, label = img[:, :, img_id], label[:, :, img_id]
                # Remove 4th dimension if multiphase
                if len(img.shape) == 4:
                    img = img[...,0] 
                img, label = self.preprocessing(img), self.preprocessing(label, isLabel=True)
            # 3D
            else:
                if len(img.shape) == 4:
                    img = img[..., 0]
                    

                img = np.expand_dims(img, axis=0)
                img = rescale(img) 
                label = np.expand_dims(label, axis=0)
                if idx % 2 == 1 and False:
                    #img = np.max(img) - img
                    #img = np.flip(img, axis=1)#flip(img)
                    label = 2 - label
                    #label = np.flip(label, axis=1)#flip(label)
                img = np.squeeze(img)
                label = np.squeeze(label)
                #np.random.seed()
                #random.seed()
                #torch.seed()
                # random flip for two clusters
                #plt.imshow(img[:, :, img.shape[2]//2], cmap='gray')
                #plt.show()
                #exit()

                if self.exp.get_from_config('rescale') is not None and self.exp.get_from_config('rescale') is True:
                    img, label = self.rescale3d(img), self.rescale3d(label, isLabel=True)
                if self.exp.get_from_config('keep_original_scale') is not None and self.exp.get_from_config('keep_original_scale'):
                    img, label = self.preprocessing3d(img), self.preprocessing3d(label, isLabel=True)  
                # Add dim to label
                if len(label.shape) == 3:
                    label = np.expand_dims(label, axis=-1)
            img_id = str(idx) + "_" + str(p_id) + "_" + str(img_id)

            img_vec = np.random.randn(self.extra_channels).astype(np.float32)#*0.1

            if True:
                if img_id.__contains__('hippocampus'):
                    img_vec = np.array([-0.05, 0.14, -0.03, 0.019]).astype(np.float32)
                elif img_id.__contains__('prostate'):
                    img_vec = np.array([-0.156, -0.25, 0.20, -0.006]).astype(np.float32)
                elif img_id.__contains__('liver'):
                    img_vec = np.array([-0.07, 0.055, -0.01, -0.087]).astype(np.float32)
            #img_vec = np.array([0.1]*self.extra_channels).astype(np.float32)#*15#np.ones(self.extra_channels).astype(np.float32)#*15
            
            if self.store:
                self.data.set_data(key=self.images_list[idx], data=(img_id, img, label, img_vec))
                img = self.data.get_data(key=self.images_list[idx])
            else:
                img = (img_id, img, label, img_vec)
           

        id, img, label, img_vec = img

        size = self.size 
        
        # Create patches from full resolution
        if self.exp.get_from_config('patchify') is not None and self.exp.get_from_config('patchify') is True and self.state == "train": 
            img, label = self.patchify(img, label) 

        if len(size) > 2:
            size = size[0:2] 

        # Normalize image
        img = np.expand_dims(img, axis=0)
        if np.sum(img) > 0:
            img = znormalisation(img)
        img = rescale(img) 
        img = img[0]

        # Merge labels -> For now single label
        label[label > 0] = 1

        # Number of defined channels
        if len(self.size) == 2:
            img = img[..., :self.exp.get_from_config('input_channels')]
            label = label[..., :self.exp.get_from_config('output_channels')]

        if id.__contains__('hippocampus'):
            cl = 'hippocampus' #0
        elif id.__contains__('prostate'):
            cl = 'prostate' #1
        elif id.__contains__('liver'):
            cl = 'liver' #2
        else:
            cl = 'unknown'

        if idx % 2 and False:
            cl = cl + "_flip"

        #print("GETITEM INDEX", self.images_list[idx])

        data_dict = {}
        data_dict['id'] = id
        data_dict['image'] = img
        data_dict['label'] = label
        data_dict['image_vec'] = img_vec
        data_dict['class'] = cl

        return data_dict#(id, img, label, img_vec)#(id, img, label, img_vec)
