import cv2
import os
from  src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
import numpy as np
import torchvision.transforms as T
import torch
import random
import matplotlib.pyplot as plt


class png_Dataset_gen(Dataset_NiiGz_3D):

    normalize = True

    def __init__(self, crop=False, buffer=True, downscale=4, extra_channels=8):
        super().__init__()
        self.crop = crop
        self.buffer = buffer
        self.downscale = downscale
        self.slice = 2
        self.extra_channels = extra_channels

    def set_vec(self, idx: str, vec) -> tuple:
        #print("INDEX", idx)
        #print("VEC", vec)
        img = self.data.get_data(key=self.images_list[idx])
        id, img, label, img_vec = img
        img_vec = vec
        self.data.set_data(key=self.images_list[idx], data=(id, img, label, img_vec))

    def set_normalize(self, normalize=True):
        self.normalize = normalize

    def load_item(self, path: str) -> np.ndarray:
        r"""Loads the data of an image of a given path.
            #Args
                path (String): The path to the nib file to be loaded."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not self.crop:
            img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        else:
            img = cv2.resize(img, dsize=(img.shape[1]//self.downscale,img.shape[0]//self.downscale), interpolation=cv2.INTER_CUBIC)

        img = cv2.convertScaleAbs(img)
        #img = img/256 
        #img = img*2 -1
        #print("MINMAX", torch.max(img), torch.min(img))
        return img

    def __getitem__(self, idx: int) -> tuple:
        r"""Standard get item function
            #Args
                idx (int): Id of item to loa
            #Returns:
                img (numpy): Image data
                label (numpy): Label data
        """

        if self.buffer:
            data = self.data.get_data(key=self.images_list[idx])

            if not data:
                img_name, p_id, img_id = self.images_list[idx]
                label_name, _, _ = self.labels_list[idx]

                
                label = self.load_item(os.path.join(self.images_path, img_name))
                
                img = np.zeros(label.shape)#np.random.uniform(size = label.shape)*0.1 -0.05  #np.ones(label.shape)*0.01#
                img[img.shape[0]//2, img.shape[1]//2, :] = 1
                img_id = str(idx) + "_" + str(img_id)
                img_vec = np.random.randn(self.extra_channels).astype(np.float32)#*0.1

                #img_vec = np.array([ 0.8374993,   0.00956093, -0.23195317,  0.8322674,  -1.,         -0.10538746]).astype(np.float32)
                
                self.data.set_data(key=self.images_list[idx], data=(img_id, img, label, img_vec))
                data = self.data.get_data(key=self.images_list[idx])
                img_id, img, label, img_vec = data
                
                data = (img_id, img, label, img_vec)
        else:
            img_name, p_id, img_id = self.images_list[idx]
            label_name, _, _ = self.labels_list[idx]

            label = self.load_item(os.path.join(self.images_path, img_name))
            img = np.zeros(label.shape)
            img_id = str(idx) + "_" + str(img_id)
            img_vec = np.random.randn(self.extra_channels).astype(np.float32)#*0.1
            data = (img_id, img, label, img_vec)

        id, img, label, img_vec = data

        if self.crop:
            pos_x = random.randint(0, img.shape[0] - self.size[0])
            pos_y = random.randint(0, img.shape[1] - self.size[1])

            img = img[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1], :]
            label = label[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1], :]

        #from matplotlib import pyplot as plt
        #plt.imshow(img[:,:,:])#outputs_fft[0, 0, :, :].real.detach().cpu().numpy())
        #plt.show()

        img = img[...,0:4]
        

        if self.normalize:
            if False:
                transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                img = torch.from_numpy(img).to(torch.float64)
                img = img.permute((2, 1, 0))
                #print(img.shape)
                img = transform(img)
                img = img.permute((2, 1, 0))
            #img = img * 2 -1
            #img = img
            label = label/256# -1#img/256/2.5 -1  #img/128 -1 #img/256/2.5 -1 #/2.5 -1


        data_dict = {}
        data_dict['id'] = id
        data_dict['image'] = img
        data_dict['label'] = label
        data_dict['image_vec'] = img_vec
        data_dict['class'] = 'None'


        return data_dict

