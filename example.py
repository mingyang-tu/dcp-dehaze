from argparse import ArgumentParser, Namespace
import numpy as np
from numpy.typing import NDArray
import cv2
import time

from dcp_dehaze import dcp_dehaze


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input image")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output image, set to None if you don't want to save the image.")
    parser.add_argument("-s", "--max_size", type=int, default=600,
                        help="Maximum size of resized image, set to 0 if not to resize.")

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

    print(f"\nImage Size: {img.shape}")

    start = time.time()

    dehaze = dcp_dehaze(
        img.astype(np.float64)
    )

    end = time.time()

    print(f"\nEllapse Time: {end - start:.4f} s")

    if args.output:
        cv2.imwrite(args.output, dehaze)
        print(f"\nSave as {args.output}")

    cv2.imshow("Original Image", img)
    cv2.imshow("Dehaze", dehaze)

    cv2.waitKey()
    cv2.destroyAllWindows()
