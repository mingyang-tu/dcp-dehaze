import numpy as np
from numpy.typing import NDArray
import cv2
from .utils.closed_form_matting import closed_form_matting_with_prior
from .utils.bilateral import cbf


def refine_tmap(img: NDArray[np.float64], tmap: NDArray[np.float64], mode: str, kwargs: dict) -> NDArray[np.float64]:
    lamb = kwargs.get("lamb", 1e-4)
    sigma_s, sigma_r = kwargs.get("sigma_s", 15.), kwargs.get("sigma_r", 0.1)
    radius, epsilon = kwargs.get("radius", 30), kwargs.get("epsilon", 0.01)

    if mode == "original":
        return bilateral_filter(soft_matting(img, tmap, lamb), sigma_s, sigma_r)
    elif mode == "soft_matting":
        return soft_matting(img, tmap, lamb)
    elif mode == "bilateral":
        return bilateral_filter(tmap, sigma_s, sigma_r)
    elif mode == "cross_bilateral":
        return cross_bilateral_filter(img, tmap, sigma_s, sigma_r)
    elif mode == "guided_filter":
        return color_guided_filter(img, tmap, radius, epsilon)
    else:
        raise ValueError("Algorithm not supported.")


def soft_matting(img: NDArray[np.float64], tmap: NDArray[np.float64], lamb: float) -> NDArray[np.float64]:
    print(f"    soft matting, lambda = {lamb}")

    prior_confidence = lamb * np.ones(tmap.shape, dtype=np.float64)
    return closed_form_matting_with_prior(img, tmap, prior_confidence)


def bilateral_filter(tmap: NDArray[np.float64], sigma_s: float, sigma_r: float) -> NDArray[np.float64]:
    print(f"    bilateral filter, sigma_s = {sigma_s}, sigma_r = {sigma_r}")

    return cv2.bilateralFilter(tmap.astype(np.float32), 0, sigma_r, sigma_s)


def cross_bilateral_filter(img: NDArray[np.float64], tmap: NDArray[np.float64],
                           sigma_s: float, sigma_r: float) -> NDArray[np.float64]:
    print(f"    cross bilateral filter, sigma_s = {sigma_s}, sigma_r = {sigma_r}")

    return cbf(img / 255., tmap, sigma_s, sigma_r)


def color_guided_filter(img: NDArray[np.float64], tmap: NDArray[np.float64],
                        radius: int, epsilon: float) -> NDArray[np.float64]:
    print(f"    guided filter, radius = {radius}, epsilon = {epsilon}")

    ROW, COL, _ = img.shape
    kernel = (radius * 2 + 1, radius * 2 + 1)

    mean_I = cv2.blur(img, kernel)
    mean_p = cv2.blur(tmap, kernel)

    corr_I = np.zeros((ROW, COL, 9), dtype=np.float64)
    for i in range(ROW):
        for j in range(COL):
            pix = np.reshape(img[i, j, :], (1, 3))
            corr_I[i, j, :] = np.ravel(np.dot(pix.T, pix))
    corr_I = cv2.blur(corr_I, kernel)

    corr_Ip = np.zeros((ROW, COL, 3), dtype=np.float64)
    for i in range(ROW):
        for j in range(COL):
            corr_Ip[i, j, :] = tmap[i, j] * img[i, j, :]
    corr_Ip = cv2.blur(corr_Ip, kernel)

    eps_U = epsilon * np.eye(3, dtype=np.float64)

    mat_a = np.zeros((ROW, COL, 3), dtype=np.float64)
    for i in range(ROW):
        for j in range(COL):
            mu = np.reshape(mean_I[i, j, :], (1, 3))
            sigma = np.reshape(corr_I[i, j, :], (3, 3)) - np.dot(mu.T, mu)
            mat_a[i, j, :] = np.dot(
                np.linalg.inv(sigma + eps_U),
                corr_Ip[i, j, :] - mean_I[i, j, :] * mean_p[i, j]
            )

    mat_b = np.zeros((ROW, COL), dtype=np.float64)
    for i in range(ROW):
        for j in range(COL):
            mat_b[i, j] = mean_p[i, j] - np.dot(mat_a[i, j, :], mean_I[i, j, :])

    mean_a = cv2.blur(mat_a, kernel)
    mean_b = cv2.blur(mat_b, kernel)

    mat_q = np.zeros((ROW, COL), dtype=np.float64)
    for i in range(ROW):
        for j in range(COL):
            mat_q[i, j] = np.dot(mean_a[i, j, :], img[i, j, :]) + mean_b[i, j]

    return mat_q
