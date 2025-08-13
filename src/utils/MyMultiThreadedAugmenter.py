
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

class MyMultiThreadedAugmenter(MultiThreadedAugmenter):
    def __len__(self):
        return len(self.generator)
    