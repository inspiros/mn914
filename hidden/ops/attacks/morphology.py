import imageio
import numpy as np
from skimage import morphology
from skimage.morphology import diamond, disk, star
import matplotlib.pyplot as plt
import argparse


def apply_erosion(img: np.ndarray,
                  struc_size: int):
    r"""
    This function is used to apply erosion morphology in grayscale image.
    - img: Input gray scale image as a numpy array.
    - struc_size: Size of the diamond structuring element.

    Returns:
    - Eroded image as a uint8 numpy array.
    """

    out_img = morphology.erosion(img, footprint=diamond(struc_size)).astype(np.uint8)
    return out_img


def apply_dilation(img: np.ndarray,
                   struc_size: int):
    r"""
    This function is used to apply dilation morphology in grayscale image.
    - img: Input gray scale image as a numpy array.
    - struc_size: Size of the disk structuring element.

    Returns:
    - Dilated image as a uint8 numpy array.
    """

    out_img = morphology.dilation(image=img, footprint=disk(struc_size)).astype(np.uint8)
    return out_img


def apply_closing(img: np.ndarray,
                  struc_size: int):
    r"""
    This function is used to apply dilation morphology in grayscale image.
    - img: Input gray scale image as a numpy array.
    - struc_size: Size of the star structuring element.

    Returns:
    -  applied image as a uint8 numpy array.
    """

    out_img = morphology.closing(image=img, footprint=star(struc_size)).astype(np.uint8)
    return out_img


def apply_opening(img: np.ndarray,
                  struc_size: int):
    r"""
    This function is used to apply dilation morphology in grayscale image.
    - img: Input gray scale image as a numpy array.
    - struc_size: Size of the dimond structuring element.

    Returns:
    -  applied image as a uint8 numpy array.
    """

    out_img = morphology.opening(image=img, footprint=diamond(struc_size))
    return out_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=None, help="direction to the image")
    parser.add_argument("--morph_type", default="erosion", help="dilation | erosion | opening | closing")
    parser.add_argument("--struc_size", type=int, default=5, help="size of structuring element")
    args = parser.parse_args()

    img = imageio.imread("../../images/3.png", pilmode='L')

    print(img.shape)

    if args.morph_type == "erosion":
        out_img = apply_erosion(img=img, struc_size=args.struc_size)
    elif args.morph_type == "dilation":
        out_img = apply_dilation(img=img, struc_size=args.struc_size)
    elif args.morph_type == "opening":
        out_img = apply_opening(img=img, struc_size=args.struc_size)
    else:
        out_img = apply_closing(img=img, struc_size=args.struc_size)
    
    print(out_img.shape)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("original image")
    plt.subplot(1, 2, 2)
    plt.imshow(out_img, cmap="gray")
    plt.title("applied image")
    plt.show()
