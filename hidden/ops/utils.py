import io

from PIL import Image

__all__ = ['encoding_quality']


def encoding_quality(
        image: Image.Image,
        quality: int = 50) -> Image.Image:
    r"""
    Changes the JPEG encoding quality level.

    Reference: `augly.image.functional.encoding_quality`

    Args:
        image (Image.Image): An instance of PIL.Image.Image.
        quality (int): JPEG encoding quality. 0 is lowest quality,
            100 is highest.
    """
    assert 0 <= quality <= 100, "'quality' must be a value in the range [0, 100]"
    src_mode = image.mode

    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    aug_image = Image.open(buffer)

    if src_mode is not None:
        aug_image = aug_image.convert(src_mode)
    return aug_image
