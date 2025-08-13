
from typing import Sequence
import torch
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

from src.models.Model_OctreeNCA_3d_patching2 import OctreeNCA3DPatch2


class MyDataParallel(torch.nn.DataParallel):
    def replicate(self, module: Any, device_ids: Sequence[int | torch.device]) -> List:
        assert isinstance(module, OctreeNCA3DPatch2)

        replicas = super().replicate(module, device_ids)
        for i, replica in enumerate(replicas):
            replica.device = device_ids[i]
            if replica.separate_models:
                for model in replica.backbone_ncas:
                    model.device = device_ids[i]
            else:
                replica.backbone_nca.device = device_ids[i]
        return replicas
