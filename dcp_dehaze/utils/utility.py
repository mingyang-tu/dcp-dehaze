import numpy as np
from numpy.typing import NDArray
import cv2


def scene_depth(tmap: NDArray[np.float64], beta: float = 0.01) -> NDArray[np.uint8]:
    depth = -np.log(tmap + 1e-8) / beta
    depth256 = np.clip(depth, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(depth256, 11)


def bgr2gray(img: NDArray[np.float64]) -> NDArray[np.float64]:
    cvt_mat = np.array([19/256, 183/256, 54/256], dtype=np.float64)
    return img.dot(cvt_mat)


def color_analysis(img: NDArray[np.float64]) -> NDArray[np.float64]:
    avgs = np.mean(img, axis=(0, 1))
    corrected = avgs[2] - avgs
    print(f"    Corrected BGR = ({corrected[0]:.3f}, {corrected[1]:.3f}, {corrected[2]:.3f})")
    return corrected
