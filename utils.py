from base64 import b64decode, b64encode
from hashlib import sha256
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def concatenate_images(image1: Image.Image, image2: Image.Image) -> Image.Image:
    # Resize the larger image to fit keep aspect ratio
    width1, height1 = image1.size
    width2, height2 = image2.size
    if height1 > height2:
        # resize larger one and keep aspect ratio
        ratio = height2 / height1
        new_width = int(width1 * ratio)
        image1 = image1.resize((new_width, height2), Image.Resampling.LANCZOS)
    else:
        ratio = height1 / height2
        new_width = int(width2 * ratio)
        image2 = image2.resize((new_width, height1), Image.Resampling.LANCZOS)
    # concatenate
    width1, height1 = image1.size
    width2, height2 = image2.size
    new_image = Image.new("RGB", (width1 + width2 + 16, height1))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (width1 + 16, 0))
    return new_image


def general_concatenate_images(
    *images: Image.Image, seperator_width: int = 16
) -> Image.Image:
    """Concatenate multiple images horizontally. Resize the larger image to fit keep
        aspect ratio.

    Args:
        *images (Image.Image): Images to concatenate.

    Returns:
        Image.Image: Concatenated image.
    """
    if len(images) == 0:
        raise ValueError("At least one image should be provided.")
    elif len(images) == 1:
        return images[0]
    else:
        new_image = images[0]
        for image in images[1:]:
            new_image = concatenate_images(new_image, image, seperator_width)
        return new_image


def image_to_base64(image: Image.Image, format: str = "PNG"):
    buffered = BytesIO()
    image.save(buffered, format=format)
    return b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_image(base64_str: str):
    image_data = b64decode(base64_str)
    return Image.open(BytesIO(image_data))


def hash_image(image: Image.Image) -> str:
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()
    return sha256(image_bytes).hexdigest()


def expand_mask(mask, expand_iteration=1, kernel_size=(3, 3)):
    return cv2.dilate(
        mask.astype(np.uint8),
        np.ones(kernel_size, dtype=np.uint8),
        iterations=expand_iteration,
    ).astype(bool)


def shrink_mask(mask, shrink_iteration=1, kernel_size=(3, 3)):
    return cv2.erode(
        mask.astype(np.uint8),
        np.ones(kernel_size, dtype=np.uint8),
        iterations=shrink_iteration,
    ).astype(bool)


def resize_image_height(image: Image.Image, size: int) -> Image.Image:
    """Resize an image to a specific height while keeping the aspect ratio.

    Args:
        image (Image.Image): Image to resize.
        size (int): Height.

    Returns:
        Image.Image: Resized image.
    """
    width, height = image.size
    new_width = int(width * size / height)
    return image.resize((new_width, size), Image.Resampling.LANCZOS)
