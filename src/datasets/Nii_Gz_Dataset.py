from cmath import nan
from os import listdir
from os.path import join
import nibabel as nib
import numpy as np
import cv2
import os
from src.datasets.Dataset_Base import Dataset_Base
import random
import matplotlib.pyplot as plt

class Nii_Gz_Dataset(Dataset_Base):
    r""".. WARNING:: Deprecated, lacks functionality of 3D counterpart. Needs to be updated to be useful again."""

    slice = -1

    def getFilesInPath(self, path: str) -> dict:
        r"""Get files in path ordered by id and slice
            #Args
                path (string): The path which should be worked through
            #Returns:
                dic (dictionary): {key:patientID, {key:sliceID, img_slice}
        """
        dir_files = listdir(join(path))
        dic = {}
    
        for f in dir_files:
            id = f
            if self.slice is not None and self.slice != -1:
                _, id, slice = f.split("_")
                if id not in dic:
                    dic[id] = {}
                dic[id][slice] = f
            else:
                if id not in dic:
                    dic[id] = {}
                dic[id][0] = (f, f, 0)    
        return dic

    def __getname__(self, idx: str) -> str:
        r"""Get name of item by id"""
        return self.images_list[idx]

    def __getitem__(self, idx: str) -> tuple:
        r"""Standard get item function
            #Args
                idx (int): Id of item to loa
            #Returns:
                img (numpy): Image data
                label (numpy): Label data
        """

        data_dict = {}
        img_id = self.__getname__(idx)
        out = self.data.get_data(key=img_id)
        if out == False:
            img_name, p_id, img_id = self.images_list[idx]
            img = nib.load(os.path.join(self.images_path, img_name)).get_fdata()
            label = nib.load(os.path.join(self.labels_path, img_name)).get_fdata()


            # Get variance path and pred path
            parent, tail = os.path.split(self.images_path)
            #print("HEADTAIL", parent, "aaa", tail)
            variance_path = os.path.join(parent, 'variance')
            pred_path = os.path.join(parent, 'pred')

            #print(variance_path, pred_path)

            if os.path.exists(os.path.join(variance_path, img_name)):
                variance = nib.load(os.path.join(variance_path, img_name)).get_fdata()
                pred = nib.load(os.path.join(pred_path, img_name)).get_fdata()
                data_dict['variance'] = np.swapaxes(variance, 0, 1)
                data_dict['pred'] = np.swapaxes(pred, 0, 1)

            #print(img.shape, label.shape, img_name)
            img, label = self.preprocessing(img, label)
            self.data.set_data(key=img_id, data=(img_id, img, label))
            out = self.data.get_data(key=img_id)

        data_dict['id'] = img_id
        data_dict['image'] = img[..., np.newaxis]
        data_dict['label'] = label
        data_dict['name'] = img_name

        #2D: data_dict['image']: HWC, data_dict['label']: HWC

        return data_dict

    def getIdentifier(self, idx: str) -> str:
        r""".. TODO:: Remove redundancy"""
        return self.__getname__(idx)

    def preprocessing(self, img: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Preprocessing of image
            #Args
                img (numpy): Image to preprocess
                label (numpy): Label to preprocess
        """
        
        #print("Preprocessing", img.shape, label.shape)

        img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        #img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #img[np.isnan(img)] = 1

        label = cv2.resize(label, dsize=self.size, interpolation=cv2.INTER_NEAREST)
        #label = np.repeat(label[:, :, np.newaxis], 3, axis=2)

        # Preprocess Labels

        if len(label.shape) == 3:
            label = label[:,:, 0:1] + label[:,:, 1:2]
        else:
            label = label[:,:, np.newaxis]

        #print(np.unique(label))
        #plt.imshow(img*80)
        #plt.show()

        #plt.imshow(label)
        #plt.show()


        #plt.imshow(label[:,:, 0])

        #label[:,:, 0] = label[:,:, 0] != 0 
        #label[:,:, 1] = 0
        #label[:,:, 2] = 0

        # REMOVE
        #label[label > 0] = 1

        return img, label

