import numpy as np
from numpy.typing import NDArray

from .dcp import get_dark_channel


def get_t(img: NDArray[np.float64], A: NDArray[np.float64], patch_size: tuple[int, int],
          mode: str, omega: float, rho: float) -> NDArray[np.float64]:
    print(f"    mode = {mode}, omega = {omega}, rho = {rho}")

    imgdivA = np.zeros(img.shape, dtype=np.float64)
    for i in range(3):
        imgdivA[:, :, i] = img[:, :, i] / A[i]

    if mode == "mul":
        tmap = 1 - omega * get_dark_channel(imgdivA, patch_size)
    elif mode == "add":
        tmap = 1 - get_dark_channel(imgdivA, patch_size) + rho
    else:
        raise ValueError("Mode not supported.")
    return tmap
