from .dcp import get_dark_channel
from .atm_light import get_A
from .transmission import get_t
from .refine_tmap import refine_tmap
from .reconstruction import reconstruct
from .utils.utility import scene_depth

import numpy as np
from numpy.typing import NDArray
import cv2


def dcp_dehaze(img: NDArray[np.float64], **kwargs) -> NDArray[np.uint8]:
    patch_size = kwargs.get("patch_size", (15, 15))

    print(f"Patch Size: {patch_size}")

    print("\nEstimating dark channel...")
    dc = get_dark_channel(img, patch_size)

    print("\nEstimating atmospheric light...")
    A = get_A(dc, img, kwargs.get("topp", 0.001))

    print("\nEstimating transmission map...")
    tmap = get_t(
        img, A, patch_size,
        kwargs.get("t_mode", "mul"),
        kwargs.get("omega", 0.9),
        kwargs.get("rho", 0.12)
    )

    print("\nTransmission map refinement...")
    tmap_ref = refine_tmap(img, tmap, kwargs.get("r_mode", "guided_filter"), kwargs)

    tmap_ref = np.clip(tmap_ref, 0, 1)

    print("\nRecovering...")
    output = reconstruct(img, A, tmap_ref, kwargs.get("t0", 0.1), kwargs.get("color_correct", False))
    output = np.clip(output, 0, 255).astype(np.uint8)

    print("\nFinish!")

    if kwargs.get("verbose", False):
        cv2.imshow("Transmission Map", np.clip(tmap_ref*255, 0, 255).astype(np.uint8))
        cv2.imshow("Scene Depth", scene_depth(tmap_ref))
        cv2.waitKey()
        cv2.destroyAllWindows()

    return output
