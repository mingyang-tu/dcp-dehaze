import numpy as np
from numpy.typing import NDArray
import cv2

from .utils.utility import bgr2gray, get_topp


def get_A(dc: NDArray[np.float64], img: NDArray[np.float64], top_p: float) -> NDArray[np.float64]:
    print(f"    top {top_p*100}%")

    dc256 = dc.astype(np.uint8)
    img_gray = bgr2gray(img)

    candidate = get_topp(dc256, top_p)

    # img_copy = img.copy()
    # img_copy[dc256 >= candidate, :] = np.array([0, 0, 255])
    # cv2.imshow(f"Top {top_p*100}%", img_copy.astype(np.uint8))
    # cv2.waitKey()

    arg_A = np.argwhere(img_gray == np.max(img_gray[dc256 >= candidate]))

    return img[arg_A[0, 0], arg_A[0, 1], :]
