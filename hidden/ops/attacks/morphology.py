import argparse

import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from skimage.morphology import ball, diamond, disk, star


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', default=None, help='path to the image')
    parser.add_argument('--struc_size', type=int, default=3, help='size of structuring element')
    args = parser.parse_args()

    img = imageio.v2.imread(args.img_path if args.img_path is not None else '../../images/3.png',
                            pilmode='L')
    print('Loaded image', img.shape)

    # visualize
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes = axes.flatten()

    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('$x$')

    axes[1].imshow(apply_erosion(img=img, struc_size=args.struc_size), cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('$\\mathrm{erosion}(x)$')

    axes[2].imshow(apply_dilation(img=img, struc_size=args.struc_size), cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('$\\mathrm{dilation}(x)$')

    axes[3].imshow(apply_opening(img=img, struc_size=args.struc_size), cmap='gray', vmin=0, vmax=255)
    axes[3].set_title('$\\mathrm{opening}(x)$')

    axes[4].imshow(apply_closing(img=img, struc_size=args.struc_size), cmap='gray', vmin=0, vmax=255)
    axes[4].set_title('$\\mathrm{closing}(x)$')

    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    plt.show()
