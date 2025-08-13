

from src.datasets.Dataset_3D import Dataset_3D
from src.datasets.Dataset_Base import Dataset_Base
import os
import numpy as np
import cv2
import torch

class Dataset_CholecSeg(Dataset_Base):
    def __init__(self):
        super().__init__()
        self.slice = None
        self.delivers_channel_axis = True
        self.is_rgb = True

        self.color_class_mapping={
                            #(127, 127, 127): 0,    #Background
                            (140, 140, 210): 1,     #Abdominal wall             (29.5%)
                            (114, 114, 255): 2,     #Liver                      (29.4%)
                            (156, 70, 231): 3,      #Gastrointestinal tract     (02.6%)
                            (75, 183, 186): 4,      #Fat                        (20.2%)
                            (0, 255, 170): 5,       #Grasper                    (03.3%)
                            (0, 85, 255): 6,        #Connective tissue          (03.1%)
                            (0, 0, 255): 7,         #Blood                      (00.5%)
                            (0, 255, 255): 8,       #Cystic duct                (00.05%)
                            (184, 255, 169): 9,     #L-hook electrocautery      (01.8%)
                            (165, 160, 255): 10,    #Gallbladder                (08.9%)
                            (128, 50, 0): 11,       #Heptatic vein              (00.02%)
                            (0, 74, 111): 12}       #Liver ligament             (00.6%)
        
        self.color_classes = [k for k in self.color_class_mapping.keys()]

        #self.labels_mapping_reverse = {}
        #for k, v in self.color_class_mapping.items():
        #    self.labels_mapping_reverse[v] = k


    
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
                dic[id][i] = f
        return dic
    
    def __getitem__(self, idx: str):
        # images have a resolution of 854x480 with 80 frames
        patient_name = self.images_list[idx][:len("videoXX")]
        first_frame = int(self.images_list[idx][len("videoXX_"):])
        path = os.path.join(self.images_path, patient_name, self.images_list[idx])


        #start_frame = np.random.randint(0, num_frames - self.size[2])
        assert self.size[2] == 80

        imgs = []
        lbls = []
        for frame in range(first_frame, first_frame + 80):
            label = cv2.imread(os.path.join(path, f"frame_{frame}_endo_color_mask.png"))
            #label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            lbls.append(label)
            image = cv2.imread(os.path.join(path, f"frame_{frame}_endo.png"))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imgs.append(image)

        imgs = np.array(imgs)
        lbls = np.array(lbls)
        #DHWC, D = time

        def reshape_batch(instack) -> np.ndarray:
            #https://stackoverflow.com/questions/65154879/using-opencv-resize-multiple-the-same-size-of-images-at-once-in-python
            N,H,W,C = instack.shape
            instack = instack.transpose((1,2,3,0)).reshape((H,W,C*N))
            outstack = cv2.resize(instack, (self.size[1], self.size[0]))
            return outstack.reshape((self.size[0], self.size[1], C, N)).transpose((3,0,1,2))

        imgs = reshape_batch(imgs)
        lbls = reshape_batch(lbls)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        imgs = imgs.astype(np.float32)
        imgs /= 255.0

        imgs -= mean
        imgs /= std


        imgs = imgs.transpose(3, 1, 2, 0)#DHWC -> CHWD
        lbls = lbls.transpose(1, 2, 0, 3)#DHWC -> HWDC

        new_new_labels = np.zeros((lbls.shape[0], lbls.shape[1], lbls.shape[2], max(self.color_class_mapping.values())), dtype=np.uint8)



        #for k, v in self.color_class_mapping.items():
        #    #k is a color
        #    mask = np.all(lbls == k, axis=-1)
        #    new_new_labels[mask, v-1] = 1
        for i, k in enumerate(self.color_classes):
            mask = np.all(lbls == k, axis=-1)
            new_new_labels[mask, i] = 1

        

        data_dict = {}
        data_dict['id'] = self.images_list[idx]
        data_dict['image'] = imgs
        data_dict['label'] = new_new_labels
        return data_dict
    
    def setPaths(self, images_path: str, images_list: str, labels_path: str, labels_list: str) -> None:
        return super().setPaths(images_path, images_list, labels_path, labels_list)