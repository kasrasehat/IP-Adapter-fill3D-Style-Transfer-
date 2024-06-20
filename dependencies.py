import os
import random
from typing import Literal, Optional

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

from minio_wrapper import MinioWrapper
from utils import resize_image_height

resolution_mode_to_value = {
    "tiny": 256,
    "low": 512,
    "medium": 768,
    "high": 1024,
}


def get_minio(env: str):
    load_dotenv()
    env_vars_prefix = env
    minio_endpoint = os.getenv(f"{env_vars_prefix}_MINIO_ENDPOINT")
    if minio_endpoint is None:
        raise ValueError("Missing environment variables MINIO_ENDPOINT")
    minio_access_key = os.getenv(f"{env_vars_prefix}_MINIO_ACCESS_KEY")
    if minio_access_key is None:
        raise ValueError("Missing environment variables MINIO_ACCESS_KEY")
    minio_secret_key = os.getenv(f"{env_vars_prefix}_MINIO_SECRET_KEY")
    if minio_secret_key is None:
        raise ValueError("Missing environment variables MINIO_SECRET_KEY")
    minio_secure = os.getenv(f"{env_vars_prefix}_MINIO_SECURE")
    if minio_secure is None:
        raise ValueError("Missing environment variables MINIO_SECURE")
    minio_secure = minio_secure == "True"
    minio = MinioWrapper(
        endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=minio_secure,
    )
    return minio


class ImageCaptioningClient:
    def __init__(
        self,
        image_captioning_api_url,
        temp_bucket_name,
        env: str = "DEV",
    ):
        self.image_captioning_api_url = image_captioning_api_url
        self.temp_bucket_name = temp_bucket_name
        self.env = env

    def get_minio_image_caption(
        self, bucket_name, object_name, env: Optional[str] = None
    ):
        if env is None:
            env = self.env
        params = {
            "object_name": object_name,
            "env": env,
            "beforebucket": bucket_name,
        }
        response = requests.post(self.image_captioning_api_url, params=params)
        response.raise_for_status()
        caption = response.json()["caption"]
        return caption

    def __call__(self, image: Image.Image):
        minio = get_minio(env=self.env)
        object_name = f"{random.randint(0, 1000)}.jpg"
        bucket_name = self.temp_bucket_name
        minio.upload_image(image, bucket_name, object_name)
        caption = self.get_minio_image_caption(bucket_name, object_name)
        return caption


class SegmentationClient:
    def __init__(
        self,
        segmentation_api_url,
        temp_bucket_name,
        env: str = "DEV",
    ):
        self.segmentation_api_url = segmentation_api_url
        self.temp_bucket_name = temp_bucket_name
        self.env = env

    def get_minio_segmentation(
        self,
        image_bucket_name: str,
        image_object_name: str,
        result_bucket_name: str,
        service_type: Literal[
            "Wall", "Floor", "Ceiling", "Furniture", "Countertop", "Backsplash"
        ] = "Furniture",
        res_mode: Literal["low", "medium", "high"] = "medium",
        env: Optional[str] = None,
    ) -> list[dict]:
        if env is None:
            env = self.env

        params = {
            "object_name": image_object_name,
            "service_type": service_type,
            "res_mode": res_mode,
            "before_bucket": image_bucket_name,
            "is_mask_output": "true",
            "after_bucket": result_bucket_name,
            "debug_mode": "false",
            "env": env,
        }

        response = requests.post(self.segmentation_api_url, params=params)
        response.raise_for_status()
        return response.json()

    def __call__(
        self,
        image: Image.Image,
        service_type: Literal[
            "Wall", "Floor", "Ceiling", "Furniture", "Countertop", "Backsplash"
        ] = "Wall",
        res_mode: Literal["low", "medium", "high"] = "medium",
        env: Optional[str] = None,
    ) -> list[dict]:
        org_width, org_height = image.size
        image = resize_image_height(image, resolution_mode_to_value[res_mode])
        minio = get_minio(env=self.env)
        object_name = f"{random.randint(0, 1000)}.jpg"
        bucket_name = self.temp_bucket_name
        minio.upload_image(image, bucket_name, object_name)
        result = self.get_minio_segmentation(
            bucket_name, object_name, bucket_name, service_type, res_mode, env
        )
        segmentations = []
        for d in result:
            label = d["tag"]
            object_name = d["object_id"] + ".png"
            rgb_mask = minio.download_image(bucket_name, object_name)
            binary_mask = np.asarray(rgb_mask)[..., 3] != 0
            # resize binary mask to original image size
            binary_mask = np.asarray(
                Image.fromarray(binary_mask).resize(
                    (org_width, org_height), Image.Resampling.LANCZOS
                )
            )
            segmentations.append({"label": label, "mask": binary_mask})
        return segmentations
