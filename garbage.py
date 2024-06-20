from dotenv import load_dotenv
import os
import skimage
import cv2
import numpy as np
from PIL import Image

def load_env_vars():
    load_dotenv()
    env_vars = {}

    segmentation_api_url = os.getenv("SEGMENTATION_API_URL")
    if segmentation_api_url is None:
        raise ValueError("Missing environment variables SEGMENTATION_API_URL")
    env_vars["segmentation_api_url"] = segmentation_api_url

    temp_bucket_name = os.getenv("TEMP_BUCKET_NAME")
    if temp_bucket_name is None:
        raise ValueError("Missing environment variables TEMP_BUCKET_NAME")
    env_vars["temp_bucket_name"] = temp_bucket_name

    image_captioning_api_url = os.getenv("IMAGE_CAPTIONING_API_URL")
    if image_captioning_api_url is None:
        raise ValueError("Missing environment variables IMAGE_CAPTIONING_API_URL")
    env_vars["image_captioning_api_url"] = image_captioning_api_url

    env = os.getenv("ENV")
    if env is None:
        raise ValueError("Missing environment variables ENV")
    env_vars["env"] = env

    io_paint_inference_api_url = os.getenv("IOPAINT_INFERENCE_API_URL")
    if io_paint_inference_api_url is None:
        raise ValueError("Missing environment variables IOPAINT_INFERENCE_API_URL")
    env_vars["iopaint_inference_api_url"] = io_paint_inference_api_url

    gradio_server_host = os.getenv("MASK_EDITOR_GRADIO_SERVER_HOST")
    if gradio_server_host is None:
        raise ValueError("Missing environment variables MASK_EDITOR_GRADIO_SERVER_HOST")
    env_vars["gradio_server_host"] = gradio_server_host

    gradio_server_port = os.getenv("MASK_EDITOR_GRADIO_SERVER_PORT")
    if gradio_server_port is None:
        raise ValueError("Missing environment variables MASK_EDITOR_GRADIO_SERVER_PORT")
    env_vars["gradio_server_port"] = int(gradio_server_port)

    return env_vars


if __name__ == "__main__":
    
    image =  cv2.imread("/home/kasra/PycharmProjects/IP_Adapter/assets/images/tmpuemnredg.PNG")[:,:,::-1]
    new_image = skimage.exposure.equalize_adapthist(image, kernel_size=None, clip_limit=1, nbins=256)
    # new_image = skimage.exposure.equalize_hist(image, nbins=256, mask=None)
    # Convert the array data type to 'uint8' (necessary for color images)
    array = image.astype(np.uint8)

    # Create a PIL Image from the ndarray
    image = Image.fromarray(array)
    image.show()

