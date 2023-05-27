import numpy as np
from numpy.typing import NDArray
import cv2

from .utils.utility import bgr2gray


def get_A(dc: NDArray[np.float64], img: NDArray[np.float64], top_p: float) -> NDArray[np.float64]:
    print(f"    top {top_p}%")

    threshold = np.percentile(dc, 100-top_p)
    candidates = dc >= threshold

    img_gray = bgr2gray(img)
    idxs = img_gray == np.max(img_gray[candidates])

    # img_copy = img.copy().astype(np.uint8)
    # img_copy[candidates, :] = np.array([0, 0, 255], dtype=img.dtype)
    # cv2.imshow(f"Top {top_p}%", img_copy)
    # cv2.waitKey()

    return np.mean(img[idxs, :], axis=0)
