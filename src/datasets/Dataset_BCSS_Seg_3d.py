

import einops
from src.datasets.Dataset_BCSS_Seg import Dataset_BCSS_Seg


class Dataset_BCSS_Seg_3d(Dataset_BCSS_Seg):
    def __init__(self, fixed_patches: bool, patch_size: tuple[int, int], images_path) -> None:
        super().__init__(fixed_patches, patch_size, images_path)
        self.slice = None

    def __getitem__(self, idx: str):
        item_dict = super().__getitem__(idx)
        # item_dict['image']: HWC, item_dict['label']: HWC

        #img: CHWD
        #lbl: HWDC

        item_dict['image'] = einops.rearrange(item_dict['image'], 'h w c -> c h w 1')
        item_dict['label'] = einops.rearrange(item_dict['label'], 'h w c -> h w 1 c')

        item_dict['image'] = einops.repeat(item_dict['image'], 'c h w d-> c h w (r d)', r=3)
        item_dict['label'] = einops.repeat(item_dict['label'], 'h w d c -> h w (r d) c', r=3)

        return item_dict