from numpy.typing import NDArray
import cv2
import time

from dcp_dehaze import dcp_dehaze


def resize(img: NDArray, max_size: int) -> NDArray:
    M = max(img.shape)
    ratio = float(max_size) / float(M)
    if M > max_size:
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)   # type: ignore
    return img


if __name__ == "__main__":

    input_path = "./test-images/test1.jpeg"

    img = cv2.imread(input_path)

    img = resize(img, 600)

    print(f"\nImage Size: {img.shape}")

    start = time.time()

    dehaze = dcp_dehaze(
        img
    )

    end = time.time()

    print(f"\nEllapse Time: {end - start:.4f} s")

    # output_path = "./result.jpeg"
    # cv2.imwrite(output_path, dehaze)
    # print(f"\nSave as {output_path}")

    cv2.imshow("Original Image", img)
    cv2.imshow("Dehaze", dehaze)

    cv2.waitKey()
    cv2.destroyAllWindows()
