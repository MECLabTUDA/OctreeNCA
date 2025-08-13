

from typing import Any
import warnings

import numpy as np


class RescaleIntensity:

    def __init__(self, out_min_max=(0,1), percentiles=(0.5, 99.5)) -> np.ndarray:
        self.out_min = out_min_max[0]
        self.out_max = out_min_max[1]
        self.percentiles = percentiles


    def __call__(self, data: np.ndarray) -> Any:
        cutoff = np.percentile(data, self.percentiles)
        np.clip(data, *cutoff, out=data)
        in_min, in_max = data.min(), data.max()
        in_range = in_max - in_min
        if in_range == 0:  # should this be compared using a tolerance?
            message = (
                f'Rescaling image  not possible'
                ' because all the intensity values are the same'
            )
            warnings.warn(message, RuntimeWarning)
            return data
        data -= in_min
        data /= in_range
        out_range = self.out_max - self.out_min
        data *= out_range
        data += self.out_min
        return data