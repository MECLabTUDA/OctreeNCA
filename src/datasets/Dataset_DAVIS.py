

from src.datasets.Dataset_3D import Dataset_3D
from src.datasets.Dataset_Base import Dataset_Base
import os
import numpy as np
import cv2

class Dataset_DAVIS(Dataset_Base):
    def __init__(self):
        super().__init__()
        self.slice = None
        self.delivers_channel_axis = True
        self.is_rgb = True
    
    def getFilesInPath(self, path: str):
        r"""Get files in path
            Args:
                path (string): The path which should be worked through
            Returns:
                dic (dictionary): {key:file_name, value: file_name}
        """
        dir_files = os.listdir(path)
        dic = {}
        for f in dir_files:
            id = f
            dic[id] = {}
            dic[id][0] = f
        return dic
    
    def __getitem__(self, idx: str):
        #low res images have a resolution of 854x480 with at least 25 and at most 104 frames
        path = os.path.join(self.images_path, self.images_list[idx])
    
        if self.state == "train":
            num_frames = len(os.listdir(path))
            frames_to_use = self.size[2]
            start_frame = np.random.randint(0, num_frames - self.size[2])
        else:
            frames_to_use = len(os.listdir(path))
            start_frame = 0
        

        imgs = []
        lbls = []
        for frame in range(start_frame, start_frame + frames_to_use):
            label = cv2.imread(os.path.join(self.labels_path, self.images_list[idx], f"{frame:05d}.png"))
            label = label[:,:,0:1]
            lbls.append(label)
            image = cv2.imread(os.path.join(path, f"{frame:05d}.jpg"))
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



        lbls[lbls > 0] = 1

        data_dict = {}
        data_dict['id'] = self.images_list[idx]
        data_dict['image'] = imgs
        data_dict['label'] = lbls
        return data_dict
    
    def setPaths(self, images_path: str, images_list: str, labels_path: str, labels_list: str) -> None:
        return super().setPaths(images_path, images_list, labels_path, labels_list)