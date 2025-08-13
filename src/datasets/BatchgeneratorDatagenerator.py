  
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
import numpy as np

from src.datasets import Dataset_Base
from src.datasets.BatchgeneratorsDataLoader import my_default_collate
from src.utils.DataAugmentations import get_transform_arr
import math
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.abstract_transforms import Compose


#class BatchgeneratorDatagenerator(SlimDataLoaderBase):
class DatasetPerEpochGenerator(SlimDataLoaderBase):
    #https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/multithreaded_with_batches.ipynb
    def __init__(self, data, num_threads_in_mt=12, batch_size=4):
        # This initializes self._data, self.batch_size and self.number_of_threads_in_multithreaded
        super(DatasetPerEpochGenerator, self).__init__(data, batch_size, num_threads_in_mt)

        self.num_restarted = 0
        self.current_position = 0
        self.was_initialized = False

    def reset(self):
        self.data_indices = np.arange(len(self._data))
        rs = np.random.RandomState(self.num_restarted)
        rs.shuffle(self.data_indices)
        self.was_initialized = True
        self.num_restarted = self.num_restarted + 1
        self.current_position = self.thread_id*self.batch_size

    def generate_train_batch(self):
        if not self.was_initialized:
            self.reset()
        idx = self.current_position
        if idx < len(self._data):
            self.current_position = idx + self.batch_size*self.number_of_threads_in_multithreaded
            indices = self.data_indices[idx: min(len(self._data),idx+self.batch_size)]
            batch = [self._data[i] for i in indices]
            batch = my_default_collate(batch)
            return batch
        else:
            self.was_initialized=False
            raise StopIteration
        
    def __len__(self):
        return math.ceil(len(self._data) / self.batch_size)

class StepsPerEpochGenerator(SlimDataLoaderBase):
    def __init__(self, data, num_steps_per_epoch:int, num_threads_in_mt=12, batch_size=4, difficulty_weighted_sampling:bool=False,
                 precomputed_difficulties: dict=None):
        # This initializes self._data, self.batch_size and self.number_of_threads_in_multithreaded
        super(StepsPerEpochGenerator, self).__init__(data, batch_size, num_threads_in_mt)
        self.num_steps_per_epoch = num_steps_per_epoch
        self.was_initialized = False
        self.difficulty_weighted_sampling = difficulty_weighted_sampling
        self._data.difficulties = precomputed_difficulties

            

    def reset(self):
        self.counter = self.thread_id
        self.was_initialized = True

    def __len__(self):
        return self.num_steps_per_epoch
    
    def generate_train_batch(self):
        if not self.was_initialized:
            self.reset()
        if self.counter >= self.num_steps_per_epoch:
            self.reset()
            raise StopIteration
        self.counter += self.number_of_threads_in_multithreaded
        if self.difficulty_weighted_sampling:
            if not hasattr(self, "normalized_weights"):
                ids = [self._data.getPublicIdByIndex(i) for i, _ in enumerate(self._data.images_list)]
                weights = np.array([self._data.difficulties[id] for id in ids])
                self.normalized_weights = weights/weights.sum()
            indices = np.random.choice(np.arange(0, len(self._data)), self.batch_size, p=self.normalized_weights)
            batch = [self._data[img_id] for img_id in indices]
        else:
            indices = np.random.choice(np.arange(len(self._data)), self.batch_size)
            batch = [self._data[i] for i in indices]
        batch = my_default_collate(batch)
        return batch
