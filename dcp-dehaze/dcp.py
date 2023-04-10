import numpy as np
from numpy.typing import NDArray
import cv2


def get_dark_channel(img: NDArray[np.float64], patch_size: tuple[int, int]) -> NDArray[np.float64]:
    kernel = np.ones(patch_size, dtype=np.uint8)
    dc = cv2.erode(np.min(img, axis=2), kernel, iterations=1)
    return dc
