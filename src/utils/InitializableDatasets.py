from src.utils.DatasetClassCreator import DatasetClassCreator
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.png_Dataset import png_Dataset
"""
This file should contain all definitios for easily loadable dataset classes. 
Loadable datasets are created like the example(s) below and can simply 
be imported from this file. 
Type hints are drawn from Dataset_Base. 

ONLY USE Dataset Classes created his way in the vizualization application. Otherwise paths wont be initialized 
and the dataset requires the existence of an experiment which will get ver unhappy if locations of models/ data are changed.
    
"""

Dataset_NiiGz_3D_loadable = DatasetClassCreator.create_loadable_dataset_class_from_class(Dataset_NiiGz_3D)
png_Dataset_Vis = DatasetClassCreator.create_loadable_dataset_class_from_class(png_Dataset)
