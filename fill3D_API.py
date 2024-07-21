import argparse
import asyncio
import datetime
import gc
import json
import logging
import os
import csv
import uuid
from io import BytesIO
from time import time
from typing import List
from PIL import Image, ImageChops
from PIL import ImageEnhance
from dotenv import load_dotenv
from controlnet_aux import CannyDetector, MLSDdetector, NormalBaeDetector
import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
import requests
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from PIL import Image
from pydantic import BaseModel, Field
import asyncio
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import time
import os
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image, ImageOps
from io import BytesIO
from transformers import pipeline
from diffusers.utils import load_image
import warnings
from utils4 import *
import uvicorn
import PIL
import gc
import logging
import argparse
import asyncio
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import time
import os
from datetime import datetime


warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(message)s')

PIL.Image.MAX_IMAGE_PIXELS = 933120000

resolution = {
    'tiny': 256,
    'low': 512,
    'medium': 768,
    'high': 1024
}

app = FastAPI()

origins = [
    # "http://192.168.1.39:1025"
    # "http://localhost",
    # "http://localhost:1025",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_row_by_name(reader, name):
    for row in reader:
        if row["Name"] == name:
            return row
    return None


def get_image(bucket_name, object_name, local_file_path, client):
    client.fget_object(bucket_name, object_name, local_file_path)
    logging.info(f"\timage downloaded: {local_file_path}")


def put_image(bucket_name, object_name, local_file_path, client):
    client.fput_object(bucket_name, object_name, local_file_path)
    logging.info(f"\timage uploaded: {object_name}")


def test_gpu_cuda():
    logging.info('test gpu and cuda:')
    logging.info('\tcuda is available: %s', torch.cuda.is_available())
    logging.info('\tdevice count: %s', torch.cuda.device_count())
    logging.info('\tcurrent device: %s', torch.cuda.current_device())
    logging.info('\tdevice: %s', torch.cuda.device(0))
    logging.info('\tdevice name: %s', torch.cuda.get_device_name())

def setup_minio(args):
    from minio import Minio

    min_client = Minio(
        args.minioserver,
        access_key=args.miniouser,
        secret_key=args.miniopass,
        secure=args.miniosecure  # Set to False if using an insecure connection (e.g., for local development)
    )
    return min_client

def runner(
            image_name,
            mask_name,
            beforebucket,
            afterbucket,
            maskbucket,
            after_name,
            prompt,
            res_mode,
            debug_mode,
            client
           ):
    
    timestamp = datetime.now().strftime("[%Y-%m-%d--%H-%M-%S]")
    logging.info(timestamp)

    ################## image
    file_name, file_extension = os.path.splitext(image_name)

    temp_save_before_path = f"./{file_name}-{timestamp}"
    local_before_path_img = temp_save_before_path + file_extension

    get_image(beforebucket, image_name, local_before_path_img, client)

    image = Image.open(local_before_path_img).convert("RGB")
    image = ImageOps.exif_transpose(image)
    logging.info(f"\timage shape: {image.size}")

    ################## mask
    file_name, file_extension = os.path.splitext(mask_name)
    temp_save_before_path = f"./{file_name}-{timestamp}"
    local_before_path_mask = temp_save_before_path + file_extension

    get_image(maskbucket, mask_name, local_before_path_mask, client)

    mask_image = Image.open(local_before_path_mask).convert("RGB")
    mask_image = ImageOps.exif_transpose(mask_image)
    logging.info(f"\tmask shape: {mask_image.size}")

    mask_image = np.array(mask_image)
    mask_image = np.where(mask_image==[120,120,205], [255,255,255], [0,0,0]) * 255
    mask_image = Image.fromarray(mask_image.astype(np.uint8)).convert("L").convert('RGB')
    image = resize_image_with_height(image, resolution[res_mode])
    main_image = image
    mask_image = resize_image_with_height(mask_image, resolution[res_mode]).resize(image.size)
    # mask_image = expand_mask(mask_image, 3)

    image, mask_image = auto_resize_to_pil(image, mask_image, logging)

    invert_target_mask = ImageChops.invert(mask_image)
    gray_target_image = image.convert('L').convert('RGB')
    gray_target_image = ImageEnhance.Brightness(gray_target_image)
    gray_target_image = gray_target_image.enhance(1)
    grayscale_img = ImageChops.darker(gray_target_image, mask_image)
    img_black_mask = ImageChops.darker(image, invert_target_mask)
    grayscale_init_img = ImageChops.lighter(img_black_mask, grayscale_img)
    init_image = grayscale_init_img
        # init_img = init_image
    mask = mask_image
    mask = resize_mask_area(mask, 1)
    # unblurred_mask = self.pipe.mask_processor.blur(mask, blur_factor=2)
    mask = pipe.mask_processor.blur(mask, blur_factor=3)

    fff = np.array(main_image)
    fff[np.array(mask)>150] = 250
    fff = Image.fromarray(fff)

    canny_image = cv2.Canny(np.array(fff), 100, 200)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image] * 3, axis=2)
    canny_image = Image.fromarray(canny_image).resize(image.size)

    normal_image = normal_processor(
        fff,
        detect_resolution=resolution[res_mode],
        image_resolution=resolution[res_mode],
    ).resize(image.size)
    
    mlsd_image = mlsd_preprocessor(
        fff,
        detect_resolution=resolution[res_mode],
        image_resolution=resolution[res_mode],
    ).resize(image.size)

    #### seg
    data = {
        "image_base64": image_to_base64(fff),
        "resolution_mode": "medium",
    }

    response = requests.post(env_vars["ADE_SEGMENTATION_MAP_API_URL"], json=data)
    response.raise_for_status()

    buffer = BytesIO(response.content)
    seg_image = Image.open(buffer).convert("RGB").resize(mlsd_image.size)
    

    logging.info(f"\tprompt: {params['Prompt']}")



    image_gen = pipe(
        params["Prompt"],
        num_inference_steps=int(params["Num Inference Steps"]),
        negative_prompt=params["Negative Prompt"],
        image=image,
        control_guidance_end = float(params["Control Gidance End"]),
        control_image=[canny_image, normal_image, mlsd_image, seg_image],
        guidance_scale=float(params["Control Guidance Scale"]),
        controlnet_conditioning_scale=[
            float(params["Canny Control"]),
            float(params["Normal Map Control"]),
            float(params["MLSD"]),
            float(params["Segmentation Control"]),
        ],
        mask_image=mask_image,
    ).images[0]

    
    os.remove(local_before_path_img)
    os.remove(local_before_path_mask)
    torch.cuda.empty_cache()
    gc.collect()

    if debug_mode:
            img_buffer = BytesIO()

            # Save the PIL image to the in-memory buffer
            image_gen.save(img_buffer, format="JPEG")

            # Set the appropriate media type for the image
            media_type = "image/jpeg"  # Adjust this based on your image format
            img_buffer.seek(0)

            return StreamingResponse(img_buffer, media_type=media_type)
    else:
            local_after_path = after_name
            local_after_path = os.path.basename(local_after_path)

            image_gen.save(local_after_path)

            # assert img.shape == enhanced_img.shape
            put_image(afterbucket, after_name, local_after_path, client)

            os.remove(local_after_path)


@app.post("/compute-ip/")
async def compute_ip(image_name: str = '',
                        mask_name: str = '',
                        prompt: str = '',
                        beforebucket: str = '',
                        afterbucket: str = '',
                        maskbucket: str = '',
                        after_name: str = '',
                        res_mode: str = 'high',
                        debug_mode: bool = False,
                        env: str = ''
                        ):
    
    try:
        with open("./mapping_services.csv", "r") as file:
            reader = csv.DictReader(file)
            params = get_row_by_name(reader, prompt)


        args.miniosecure = bool(os.getenv(f'{env}_MINIO_SECURE'))
        args.miniouser = os.getenv(f'{env}_MINIO_ACCESS_KEY')
        args.miniopass = os.getenv(f'{env}_MINIO_SECRET_KEY')
        args.minioserver = os.getenv(f'{env}_MINIO_ADDRESS')

        # args.minioserver = "192.168.32.33:9000"
        # args.miniouser = "test_user_chohfahe7e"
        # args.miniopass = "ox2ahheevahfaicein5rooyahze4Zeidung3aita6iaNahXu"
        # args.miniosecure = False       

        client = setup_minio(args)

        logging.info(f'\tstart inference')
        logging.info(f'\tresolution mode: {res_mode}')
        start = time.time()

    
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            runner,
            image_name,
            mask_name,
            beforebucket,
            afterbucket,
            maskbucket,
            after_name,
            prompt.strip(),
            res_mode,
            debug_mode,
            client
            )
        
        end = time.time()

        logging.info(f'\tdone, time: {round(end - start, 4)}')
        logging.info("\tINFO: POST /compute-ip HTTP/1.1 200 OK")

        return response
    
    except Exception as e:
        torch.cuda.empty_cache()
        logging.error(f'\tERROR: /compute-ip HTTP:/500, {e}')
        raise HTTPException(status_code=500, detail=str(e))



def load_env_vars():
    load_dotenv()
    env_vars = {}

    segmentation_api_url = os.getenv("ADE_SEGMENTATION_MAP_API_URL")
    if segmentation_api_url is None:
        raise ValueError("Missing environment variables ADE_SEGMENTATION_MAP_API_URL")
    env_vars["ADE_SEGMENTATION_MAP_API_URL"] = segmentation_api_url
    
    return env_vars


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Setup an controlnet API for ID service."
    )

    parser.add_argument("--server", "-s", type=str, help="Host ip address.")
    parser.add_argument("-port", "-p", type=int, help="Port number.")
    parser.add_argument(
        "--checkpoint", "-c", help="Diffusers checkpoint.", default="/home/kasra/PycharmProjects/IP_Adapter/inpaint_sdxl_model" , required=False
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        "-e",
        help="Enable Model CPU Offload",
        action="store_true",
        default=True,
        required=False,
    )

    args = parser.parse_args()

    args.server = "127.0.0.1"
    args.port = 1025
    # args.checkpoint = "/media/pourmirzaei/SSD/models/inpainting/v4.3.0"

    test_gpu_cuda()

    global pipe, mlsd_preprocessor, normal_processor, controlnet_canny, env_vars
    pipe = prepare_model_from_diffusers_controlnet(args, logging)
    mlsd_preprocessor = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
    normal_processor = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_normalbae")
    controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
        
    env_vars = load_env_vars()

    uvicorn.run(app, host=args.server, port=args.port)


