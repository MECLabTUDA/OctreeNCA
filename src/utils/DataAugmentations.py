import batchgenerators.transforms.spatial_transforms as spatial_transforms
import batchgenerators.transforms.noise_transforms as noise_transforms
import batchgenerators.transforms.color_transforms as color_transforms
import batchgenerators.transforms.resample_transforms as resample_transforms
import batchgenerators.transforms.abstract_transforms as abstract_transforms
import batchgenerators.transforms.utility_transforms as utility_transforms


def get_transform_arr():
    #https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/nnunet/training/data_augmentation/data_augmentation_moreDA.py#L41
    transform = []
    transform.append(utility_transforms.RenameTransform('label', 'target', True))
    transform.append(utility_transforms.RenameTransform('image', 'data', True))
    #self.transform.append(spatial_transforms.SpatialTransform()) #TODO params
    transform.append(noise_transforms.GaussianNoiseTransform(p_per_sample=0.1))
    transform.append(noise_transforms.GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                        p_per_channel=0.5))
    transform.append(color_transforms.BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    #maybe additive brightness
    transform.append(color_transforms.ContrastAugmentationTransform(p_per_sample=0.15))
    transform.append(resample_transforms.SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                p_per_channel=0.5,
                                                order_downsample=0, order_upsample=3, p_per_sample=0.25))
    # TODO add gamma transform
    # TODO add mirror transform

    transform.append(utility_transforms.RenameTransform('target', 'label', True))
    transform.append(utility_transforms.RenameTransform('data', 'image', True))
    return transform



def get_augmentation_dataset(dataset_class):
    class DataAugmentationDataset(dataset_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.transform = abstract_transforms.Compose(get_transform_arr())
        def __getitem__(self, idx):
            d_dict = super().__getitem__(idx)

            if self.state == 'train':
                # apply augmentations
                d_dict = self.transform.__call__(**d_dict)


            return d_dict
        

    return DataAugmentationDataset
