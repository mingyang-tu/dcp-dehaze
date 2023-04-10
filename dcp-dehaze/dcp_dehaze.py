from .dcp import get_dark_channel
from .atm_light import get_A
from .transmission import get_t
from .refine_tmap import refine_tmap
from .reconstruction import reconstruct

import numpy as np
from numpy.typing import NDArray


def dcp_dehaze(img: NDArray[np.float64], **kwargs) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    patch_size = getattr(kwargs, "patch_size", (15, 15))

    print("\nEstimating dark channel...")
    dc = get_dark_channel(img, patch_size)

    print("\nEstimating atmospheric light...")
    A = get_A(dc, img, getattr(kwargs, "topp", 0.001))

    print("\nEstimating transmission map...")
    tmap = get_t(img, A, patch_size,
                 getattr(kwargs, "t_mode", "mul"),
                 getattr(kwargs, "omega", 0.9),
                 getattr(kwargs, "rho", 0.12)
                 )

    print("\nTransmission map refinement...")
    tmap_ref = refine_tmap(img, tmap, getattr(kwargs, "r_mode", "default"), kwargs)

    tmap_ref = np.clip(tmap_ref, 0, 1)

    print("\nRecovering...")
    output = reconstruct(img, A, tmap_ref, getattr(kwargs, "t0", 0.1), getattr(kwargs, "color_correct", False))

    print("\nFinish!")

    return output, tmap_ref
