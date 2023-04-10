from .dcp import get_dark_channel
from .atm_light import get_A
from .transmission import get_t
from .refine_tmap import refine_tmap
from .reconstruction import reconstruct

import numpy as np
from numpy.typing import NDArray
from argparse import Namespace


def dcp_dehaze(img: NDArray[np.float64], patch_size: tuple[int, int], args: Namespace) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    assert patch_size[0] == patch_size[1]

    print("\nEstimating dark channel...")
    dc = get_dark_channel(img, patch_size)

    print("\nEstimating atmospheric light...")
    A = get_A(dc, img)

    print("\nEstimating transmission map...")
    tmap = get_t(img, A, patch_size, mode=args.t_mode)

    print("\nTransmission map refinement...")
    tmap_ref = refine_tmap(img, tmap, mode=args.refine_alg)

    tmap_ref = np.clip(tmap_ref, 0, 1)

    print("\nRecovering...")
    output = reconstruct(img, A, tmap_ref, color_correct=args.color_correct)

    print("\nFinish!")

    return output, tmap_ref
