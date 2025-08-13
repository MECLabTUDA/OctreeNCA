import numpy as np

class ZNormalization:

    def __init__(self) -> None:
        pass

    def __call__(self, data: np.ndarray) -> np.ndarray:
        mean, std = data.mean(), data.std()
        if std == 0:
            message = (
                'Standard deviation is 0 for masked values'
                f' in image "" ()'
            )
            raise RuntimeError(message)
        data -= mean
        data /= std
        return data