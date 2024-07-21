import os
from base64 import b64decode, b64encode
from hashlib import sha256
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionInpaintPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image
from torchvision import transforms


def plot_images(image1, image2, image3, figure_title):
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plot the first image on the first subplot
    axes[0].imshow(image1)
    axes[0].set_title("Original")

    # Plot the second image on the second subplot
    axes[1].imshow(image2)
    axes[1].set_title("Mask")

    # Plot the third image on the third subplot
    axes[2].imshow(image3)
    axes[2].set_title("Edited")

    # Remove axis labels and ticks
    for ax in axes:
        ax.axis("off")
    fig.suptitle(figure_title)
    # Display the figure
    plt.show()


def resize_image_with_height(pil_image, new_height):
    # Calculate the aspect ratio
    width, height = pil_image.size
    aspect_ratio = width / height

    # Calculate the new width based on the aspect ratio
    new_width = int(new_height * aspect_ratio)

    # Resize the image while preserving the aspect ratio
    resized_image = pil_image.resize((new_width, new_height))

    # Return the resized image
    return resized_image


def get_concat_v(im1, im2):
    dst = Image.new("RGB", (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def normal_provider(image, depth_estimator, normal_size):
    image = depth_estimator(image)["predicted_depth"][0]

    image = image.numpy()

    image_depth = image.copy()
    image_depth -= np.min(image_depth)
    image_depth /= np.max(image_depth)

    bg_threhold = 0.05

    x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    x[image_depth < bg_threhold] = 0

    # image_array = np.array(x)
    # plt.imshow(image_array, cmap='gray')
    # plt.show()

    y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    y[image_depth < bg_threhold] = 0

    z = np.ones_like(x) * np.pi * 2.0

    image = np.stack([x, y, z], axis=2)
    image /= np.sum(image**2.0, axis=2, keepdims=True) ** 0.5
    image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    controlnet_conditioning_image = Image.fromarray(image)

    controlnet_conditioning_image = controlnet_conditioning_image.resize(normal_size)
    return controlnet_conditioning_image


def expand_mask(sel_mask, expand_iteration=1):
    sel_mask = np.array(sel_mask)
    expand_iteration = int(np.clip(expand_iteration, 1, 100))
    new_sel_mask = cv2.dilate(
        sel_mask, np.ones((3, 3), dtype=np.uint8), iterations=expand_iteration
    )
    return Image.fromarray(new_sel_mask)


def auto_resize_to_pil(init_image, mask_image, logging):
    assert (
        init_image.size == mask_image.size
    ), "The sizes of the image and mask do not match"
    width, height = init_image.size
    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    # if new_width < width or new_height < height:
    # if (new_width / width) < (new_height / height):
    # scale = new_height / height
    # else:
    # scale = new_width / width
    # resize_height = int(height*scale+0.5)
    # resize_width = int(width*scale+0.5)
    # if height != resize_height or width != resize_width:
    # logging.info(f"resize: ({height}, {width}) -> ({new_height}, {new_width})")
    init_image = transforms.functional.resize(
        init_image, (new_height, new_width), transforms.InterpolationMode.LANCZOS
    )
    mask_image = transforms.functional.resize(
        mask_image, (new_height, new_width), transforms.InterpolationMode.LANCZOS
    )
    # if resize_height != new_height or resize_width != new_width:
    #     logging.info(f"center_crop: ({resize_height}, {resize_width}) -> ({new_height}, {new_width})")
    #     init_image = transforms.functional.center_crop(init_image, (new_height, new_width))
    #     mask_image = transforms.functional.center_crop(mask_image, (new_height, new_width))
    return init_image, mask_image


def prepare_model_from_diffusers(args, logging):
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        args.checkpoint, torch_dtype=torch.float16
    ).to("cuda")

    pipeline.unet.to(memory_format=torch.channels_last)
    logging.info(f"use channels last memory format for unet module")

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

    if args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
        logging.info(f"enable model cpu offloading")
    else:
        pipeline.to("cuda")
        logging.info(f"enable cuda mode")

    pipeline.set_progress_bar_config(disable=True)

    return pipeline


def prepare_model_from_diffusers_controlnet(args, logging):
    #### mlsd
    controlnet_mlsd = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
    )

    #### normal
    controlnet_normal = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_normalbae", torch_dtype=torch.float16
    )

    #### canny
    controlnet_canny = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
    )
    #### segmentation
    controlnet_seg = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16
    )

    pipeline = StableDiffusionControlNetInpaintPipeline.from_single_file(
        os.path.join(
            args.checkpoint, "realisticVisionV60B1_v60B1InpaintingVAE.safetensors"
        ),
        controlnet=[
            controlnet_canny,
            controlnet_normal,
            controlnet_mlsd,
            controlnet_seg,
        ],
        torch_dtype=torch.float16,
        variant="fp16",
    )

    pipeline.unet.to(memory_format=torch.channels_last)
    logging.info(f"use channels last memory format for unet module")

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

    if args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
        logging.info(f"enable model cpu offloading")
    else:
        pipeline.to("cuda")
        logging.info(f"enable cuda mode")

    pipeline.set_progress_bar_config(disable=True)

    return pipeline


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


def resize_mask_area(self, mask, scale):
        """
        Resize the non-black area of the mask while maintaining its shape and aspect ratio.

        Parameters:
        mask (numpy.ndarray): The input mask to resize.
        percentage (float): The percentage to resize the mask area (e.g., 50 for 50%).

        Returns:
        numpy.ndarray: The mask with resized non-black area.
        """
        # Convert numpy mask to PIL image
        mask_pil = mask
        
        # Find the bounding box of the non-black area
        bbox = mask_pil.getbbox()
        x, y, w, h = bbox
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        h = y1 - y0
        

        #     # Draw the bounding box on the mask_pil image
        # draw = ImageDraw.Draw(mask_pil)
        # draw.rectangle(bbox, outline=255, width=2)
        # Extract the non-black area
        mask_area = mask_pil.crop(bbox)
        
        # Calculate the new size
        new_width = int(mask_area.width * scale)
        new_height = int(mask_area.height * scale)
        new_height = new_height - new_height % 8
        new_width = new_width - new_width % 8
        
        # Resize the non-black area
        resized_area = mask_area.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create a new mask with the same shape as the original
        new_mask_pil = Image.new('RGB', mask_pil.size)
        
        # Calculate the new top-left corner position to place the resized area back
        new_x = x0 + (w - new_width) // 2
        new_y = y0 + (h - new_height) // 2
        
        # Ensure the new position is within the mask bounds
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_x_end = min(mask_pil.width, new_x + new_width)
        new_y_end = min(mask_pil.height, new_y + new_height)
        
        # Place the resized area back into the new mask
        new_mask_pil.paste(resized_area, (new_x, new_y, new_x_end, new_y_end))
        
        return new_mask_pil