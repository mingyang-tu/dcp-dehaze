from argparse import ArgumentParser, Namespace
import numpy as np
from numpy.typing import NDArray
import cv2
import time

from lib import dcp_dehaze, scene_depth


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input image")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output image, set to None if you don't want to save the image.")
    parser.add_argument("-s", "--max_size", type=int, default=600,
                        help="Maximum size of resized image, set to 0 if not to resize.")
    parser.add_argument("-p", "--patch_size", type=int, default=15,
                        help="Patch size while calculating dark channel.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show the results or not.")

    parser.add_argument("--t_mode", type=str, default="mul",
                        help="Options: [\"mul\", \"add\"]")
    parser.add_argument("--refine_alg", type=str, default="default",
                        help="Options: [\"default\", \"soft_matting\", \"bilateral\", \"cross_bilateral\", \"guided_filter\"]")
    parser.add_argument("--color_correct", action="store_true")

    args = parser.parse_args()
    return args


def resize(img: NDArray, max_size: int) -> NDArray:
    M = max(img.shape)
    ratio = float(max_size) / float(M)
    if M > max_size:
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)   # type: ignore
    return img


if __name__ == "__main__":
    args = parse_args()
    img = cv2.imread(args.input)

    if args.max_size > 0:
        img = resize(img, args.max_size)

    img = img.astype(np.float64)

    patch_size = (args.patch_size, args.patch_size)

    print(f"\nImage Size: {img.shape}")
    print(f"Patch Size: {patch_size}")

    start = time.time()

    dehaze, tmap = dcp_dehaze(img, patch_size, args)

    end = time.time()

    print(f"\nEllapse Time: {end - start:.4f} s")

    dehaze = np.clip(dehaze, 0, 255).astype(np.uint8)

    if args.output:
        cv2.imwrite(args.output, dehaze)
        print(f"\nSave as {args.output}")

    if args.verbose:
        cv2.imshow("Original Image", img.astype(np.uint8))
        cv2.imshow("Dehaze", dehaze)
        cv2.imshow("Transmission Map", np.clip(tmap*255, 0, 255).astype(np.uint8))
        cv2.imshow("Scene Depth", scene_depth(tmap))

        cv2.waitKey()
        cv2.destroyAllWindows()
