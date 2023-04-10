import numpy as np
from numpy.typing import NDArray
import math


def cbf(guide: NDArray[np.float64], target: NDArray[np.float64],
        sigma_s: float, sigma_r: float) -> NDArray[np.float64]:
    ROW, COL, _ = guide.shape
    radius = math.ceil(sigma_s * 2)

    result = np.zeros((ROW, COL), dtype=np.float64)

    for i in range(ROW):
        for j in range(COL):
            x1, x2 = max(i - radius, 0), min(i + radius + 1, ROW)
            y1, y2 = max(j - radius, 0), min(j + radius + 1, COL)
            filt = space_gaussian((x1-i, x2-i), (y1-j, y2-j), sigma_s) * \
                color_gaussian(guide[x1: x2, y1: y2, :], guide[i, j, :], sigma_r)
            filt /= np.sum(filt)
            result[i, j] = np.sum(filt * target[x1:x2, y1:y2])
    return result


def space_gaussian(x_range: tuple[int, int], y_range: tuple[int, int], sigma_s: float) -> NDArray[np.float64]:
    x1, x2 = x_range
    y1, y2 = y_range
    row_i, col_i = np.mgrid[x1: x2, y1: y2]
    return np.exp(- (row_i ** 2 + col_i ** 2) / (2 * sigma_s**2))


def color_gaussian(mat: NDArray[np.float64], center: NDArray[np.float64], sigma_r: float) -> NDArray[np.float64]:
    ROW, COL, LEN = mat.shape
    diff = np.zeros((ROW, COL), dtype=np.float64)
    for i in range(LEN):
        diff += (mat[:, :, i] - center[i]) ** 2
    return np.exp(- diff / (2 * sigma_r**2))
