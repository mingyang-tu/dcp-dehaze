import numpy as np
from numpy.typing import NDArray
from .utils.utility import color_analysis


def reconstruct(img: NDArray[np.float64], A: NDArray[np.float64], tmap: NDArray[np.float64],
                t0: float, color_correct: bool) -> NDArray[np.float64]:
    print(f"    t0 = {t0}, color_correct = {color_correct}")

    if color_correct:
        A_c = A - color_analysis(img)
    else:
        A_c = A

    img_J = np.zeros(img.shape, dtype=np.float64)
    nonzero_t = np.maximum(tmap, t0)
    for i in range(3):
        img_J[:, :, i] = (img[:, :, i] - A_c[i]) / nonzero_t + A_c[i]

    return img_J
