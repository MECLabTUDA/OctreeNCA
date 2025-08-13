"""
import batchgenerators.dataloading.multi_threaded_augmenter as multi_threaded_augmenter
import batchgenerators.dataloading.data_loader as data_loader

class __DataLoader(data_loader.SlimDataLoaderBase):
    def __init__(self, data, batch_size, number_of_threads_in_multithreaded=None):
        #super().__init__(data, batch_size, number_of_threads_in_multithreaded)


class BatchgeneratorsDataLoader(multi_threaded_augmenter.MultiThreadedAugmenter):
    def __init__(self, data_set, num_processes, num_cached_per_queue=2, seeds=None, pin_memory=False, timeout=10, wait_time=0.02):





        super().__init__(_data_loader, None, num_processes, num_cached_per_queue, seeds, pin_memory, timeout, wait_time)
"""