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
import random

class Dataset_BCSS(Dataset_Base):

    def __init__(self):
        super().__init__()
        self.slice = 0
        self.color_label_dic = {
            # id, category
            0: 0,
            1: 1, 
            2: 2,
            3: 3,
            4: 4, 
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
            10: 10,
            11: 11, 
            12: 12,
            13: 13,
            14: 14, 
            15: 15,
            16: 16,
            17: 17,
            18: 18,
            19: 19,
            20: 20,
            21: 21, 
            22: 22,
            23: 23,
            24: 24, 
            25: 25,
            26: 26,
            27: 27,
            28: 28,
            29: 29,
        }
        self.color_label_dic = {
            # id, category
            1: 0, 
            2: 1,
            3: 2,
            4: 3,
            0: 4,
        }

    def getFilesInPath(self, path):
        r"""Get files in path
            Args:
                path (string): The path which should be worked through
            Returns:
                dic (dictionary): {key:file_name, value: file_name}
        """
        dir_files = listdir(join(path))
        dic = {}
        for f in dir_files:
            id = f[:-4]
            dic[id] = {}
            dic[id][0] = f
        return dic

    def __getname__(self, idx):
        r"""Get name of item by id"""
        return self.images_list[idx]

    def __getitem__(self, idx):
        r"""Standard get item function
            Args:
                idx (int): Id of item to loa
            Returns:
                img (numpy): Image data
                label (numpy): Label data
        """

        img_id = self.__getname__(idx)
        out = self.data.get_data(key=img_id)
        if not out:
            #img = cv2.imread(os.path.join(self.images_path, self.images_list[idx]))
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #label = cv2.imread(os.path.join(self.labels_path, self.images_list[idx][:-4] + ".png"))
            #label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            img = Image.open(os.path.join(self.images_path, self.images_list[idx]))
            label = 0 #Image.open(os.path.join(self.labels_path, self.images_list[idx]))
        

            img, label = self.preprocessing(img, label)


            img_id = "_" + str(img_id)[:-4].replace("_", "") + "_0"
            self.data.set_data(key=img_id, data=(img_id, img, label))

            print("LOAD DATA")
            out = self.data.get_data(key=img_id)

        # Random tile

        img_id, img, label = out

        pos_x = random.randint(0, img.shape[0] - self.size[0])
        pos_y = random.randint(0, img.shape[1] - self.size[1])

        img2 = img[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1], :]
        #label2 = label[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1], :]
        
        #from matplotlib import pyplot as plt
        #plt.imshow(img2[:,:,0])#outputs_fft[0, 0, :, :].real.detach().cpu().numpy())
        #plt.show()
        #if train == False:
        #    img2 = img
        #    label2 = label

        return img_id, img2, img2  
    
    def preprocessing(self, img, label):
        r"""Preprocessing of image
            Args:
                img (numpy): Image to preprocess
                label (numpy): Label to preprocess
        """
        
        # RESCALE
        #width = int(img.size[0]/4)
        #height = int(img.size[1]/4)
        #img = np.array(img.resize((width, height), Image.ANTIALIAS))
        #label = np.array(label.resize((width, height), Image.NEAREST))

        #label = np.stack((label, label, label), axis=2)

        img = (np.array(img)/128) -1

        # Randomly Choose Tile
        pos_x = random.randint(0, img.shape[0] - self.size[0])
        pos_y = random.randint(0, img.shape[1] - self.size[1])

        img_loc = img[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1], :]
            #label_loc = label[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1], :]
            #if len(np.unique(label_loc)) > 1:
            #    break

        #img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #unique_labels = np.unique(label)#np.unique(label, axis=2)
        #label_mask = np.zeros((label.shape[0], label.shape[1], 5))

        #for ul in unique_labels:
        #    #ul = ul.tolist()
        #    if ul not in self.color_label_dic:
        #        continue
        #    label_id = int(self.color_label_dic[ul])
        #    mask = label == ul #np.all(label == ul, axis=-1)
        #    #print(mask.shape)
        #    label_mask[mask, label_id] = 1

        data_dict = {}
        data_dict['image'] = img_loc
        data_dict['label'] = img_loc

        return data_dict
   
        #print(unique_labels)
