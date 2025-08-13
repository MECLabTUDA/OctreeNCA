
import numpy as np
import collections, inspect, math, random, logging
from collections import OrderedDict

#from batchgenerators.dataloading.data_loader import default_collate
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.abstract_transforms import Compose

from src.utils.DataAugmentations import get_transform_arr

def my_default_collate(batch):
        '''
        heavily inspired by the default_collate function of pytorch
        :param batch:
        :return:
        '''
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (int, np.int64)):
            return np.array(batch).astype(np.int32)
        elif isinstance(batch[0], (float, np.float32)):
            return np.array(batch).astype(np.float32)
        elif isinstance(batch[0], (np.float64,)):
            return np.array(batch).astype(np.float64)
        elif isinstance(batch[0], (dict, OrderedDict)):
            return {key: my_default_collate([d[key] for d in batch]) for key in batch[0]}
        elif isinstance(batch[0], (tuple, list)):
            transposed = zip(*batch)
            return [my_default_collate(samples) for samples in transposed]
        elif isinstance(batch[0], str):
            return batch
        else:
            raise TypeError('unknown type for batch:', type(batch))
        

def get_batchgenerators_dataloader_dataset(dataset_class, augmentations:bool, num_steps_per_epoch: int,
                                           batch_size: int, num_workers: int):
    logging.basicConfig(level=logging.DEBUG)
    
        

    class MyMultiThreadedAugmenter(MultiThreadedAugmenter):
        counter_per_epoch = 0
        def __len__(self):
            if num_steps_per_epoch is None:
                return math.ceil(len(self.generator) / batch_size)
            return num_steps_per_epoch

        def __next__(self):
            item = super().__next__()
            self.counter_per_epoch += 1
            if self.counter_per_epoch > len(self):
                self.counter_per_epoch = 0
                raise StopIteration
            return item


    class BatchgeneratorsDataLoaderDataset(dataset_class):
        #class BatchgeneratorsDataLoaderDatasetDataSetIterator(collections.Iterator):
        #    pass

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if augmentations:
                transform = get_transform_arr()
            else:
                transform = []
            transform.append(NumpyToTensor(keys=['image', 'label']))

            self.numpy_to_tensor = NumpyToTensor(keys=['image', 'label'])
            self.thread_id = 0
            self.internal_index = 0
            self.multithreaded_augmentor = MyMultiThreadedAugmenter(self, transform=Compose(transform), num_processes=num_workers)

            if num_steps_per_epoch is None:
                self.num_restarted = 0
                self.current_position = 0
                self.was_initialized = False

        def __reset(self):
            assert num_steps_per_epoch is None
            self.available_indices = np.arange(len(self))
            rs = np.random.RandomState(self.num_restarted)
            rs.shuffle(self.available_indices)
            self.was_initialized = True
            self.num_restarted = self.num_restarted + 1
            self.current_position = self.thread_id*batch_size
            

        def set_thread_id(self, thread_id):
            self.thread_id = thread_id

        def setState(self, state: str) -> None:
            super().setState(state)
            self.reinitialize()

        def setPaths(self, images_path: str, images_list: str, labels_path: str, labels_list: str):
            super().setPaths(images_path, images_list, labels_path, labels_list)
            self.reinitialize()

        def reinitialize(self):
            self.internal_index = 0
            self.multithreaded_augmentor.restart()
            if num_steps_per_epoch is None:
                self.__reset()

        def __iter__(self):
            return self.multithreaded_augmentor

        def __getitem__(self, idx, convert_to_torch=True):
            d_dict = super().__getitem__(idx)
            if convert_to_torch:
                d_dict = self.numpy_to_tensor(**d_dict)

            return d_dict

        def __next__(self):
            if self.state == 'test':
                item = self.__getitem__(self.internal_index, False)
                self.internal_index += 1
                return my_default_collate([item])

            if num_steps_per_epoch is not None:
                batch_indices = np.random.randint(0, len(self), size=batch_size)
            else:
                if not self.was_initialized:
                    self.__reset()
                idx = self.current_position
                if idx < len(self):
                    self.current_position = idx + batch_size*num_workers
                    batch_indices = self.available_indices[idx:min(len(self),idx+batch_size)]
                else:
                    self.was_initialized = False
                    raise StopIteration



            batch = [self.__getitem__(i, False) for i in batch_indices]
            batch = my_default_collate(batch)
            
            return batch
    return BatchgeneratorsDataLoaderDataset