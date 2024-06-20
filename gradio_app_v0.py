from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler, AutoencoderKL, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from PIL import Image
from transformers import pipeline
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from ada_pallete import ada_palette
from controlnet_aux import NormalBaeDetector
from ip_adapter import IPAdapter, IPAdapterPlus, IPAdapterPlusXL
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from rembg import remove
from PIL import Image
import torch
from ip_adapter import IPAdapterXL
from ip_adapter.utils import register_cross_attention_hook, get_net_attn_map, attnmaps2images
from PIL import Image, ImageChops
from PIL import ImageEnhance
import numpy as np
import glob
import os
from io import BytesIO
from typing import Literal
import torch
import gradio as gr
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from transformers import pipeline
from diffusers import AutoPipelineForInpainting, DEISMultistepScheduler, StableDiffusionInpaintPipeline, AutoencoderKL
from transformers import BlipProcessor, BlipForConditionalGeneration
import gc
from dotenv import load_dotenv
import os


class Generator:
    def __init__(self):
        base_model_path = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        device = "cuda"
        image_encoder_path = "models/image_encoder/"
        # ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
        ip_ckpt = "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
        # models/ip-adapter-plus_sd15.bin
        # models/ip-adapter_sd15.bin
        device = "cuda"
        torch.cuda.empty_cache()
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        self.depth_estimator = pipeline('depth-estimation')
        controlnet_canny = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0", variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
        checkpoint_depth = "diffusers/controlnet-depth-sdxl-1.0"
        controlnet_depth = ControlNetModel.from_pretrained(checkpoint_depth, variant="fp16", use_safetensors=True,
                                                           torch_dtype=torch.float16).to(device)
        controlnet = [controlnet_canny, controlnet_depth]
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            use_safetensors=True,
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(device)
        self.pipe.unet = register_cross_attention_hook(self.pipe.unet)
        self.ip_model = IPAdapterPlusXL(self.pipe, image_encoder_path, ip_ckpt, device="cuda", num_tokens=16)

    def canny_image(self, image):
        image1 = np.array(image)

        low_threshold = 50
        high_threshold = 150

        image1 = cv2.Canny(image1, low_threshold, high_threshold)
        image1 = image1[:, :, None]
        image1 = np.concatenate([image1, image1, image1], axis=2)
        control_image = Image.fromarray(image1).resize(image.size)
        return control_image

    def depth_image(self, image: Image.Image):
        depth_image = self.depth_estimator(image)['depth']
        np_image = (np.array(depth_image) / 256).astype('uint8')
        depth_map = self.new_method(image, np_image)
        return depth_map

    def new_method(self, image, np_image):
        depth_map = Image.fromarray(np_image).resize(image.size)
        return depth_map

    def Generate_caption(self, raw_image):
        # self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        # self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

        # unconditional image captioning
        text = "image of"
        inputs = self.processor(raw_image, text, return_tensors="pt")

        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)

    def __call__(
            self,
            object_type,
            init_image,
            mask_image,
            reference_image,
            style_type,
            brightness,
            canny_scale,
            depth_scale,
            canny_endpoint,
            depth_endpoint,
            num_inference_step,
            strength,
            scale,
            guidance,
            sampler_type,
    ):
        torch.cuda.empty_cache()
        gc.collect()
        reference_image = reference_image.resize(init_image.size, Image.Resampling.LANCZOS)
        mask_image = mask_image.convert('L').convert('RGB').resize(init_image.size, Image.Resampling.LANCZOS)
        mask_target_img = ImageChops.lighter(init_image, mask_image)
        invert_target_mask = ImageChops.invert(mask_image)
        gray_target_image = init_image.convert('L').convert('RGB')
        gray_target_image = ImageEnhance.Brightness(gray_target_image)
        gray_target_image = gray_target_image.enhance(brightness)
        grayscale_img = ImageChops.darker(gray_target_image, mask_image)
        img_black_mask = ImageChops.darker(init_image, invert_target_mask)
        grayscale_init_img = ImageChops.lighter(img_black_mask, grayscale_img)
        init_image = grayscale_init_img
        init_img = init_image
        mask = mask_image.resize(init_image.size)
        if sampler_type == "UniPCMultistepScheduler":
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_type == "DPMSolverMultistepScheduler":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_type == "DEISMultistepScheduler":
            self.pipe.scheduler = DEISMultistepScheduler.from_config(self.pipe.scheduler.config)
        prompt = self.Generate_caption(reference_image)
        prompt = object_type + 'covered by' + prompt.split('of')[-1]

        controlnet_conditioning_scale = [canny_scale, depth_scale]
        control_guidance_start = [0, 0]
        control_guidance_end = [canny_endpoint, depth_endpoint]
        canny_control_image_init = self.canny_image(init_image)
        depth_control_image = self.depth_image(init_image)
        if style_type == 'only_style':
            images = self.ip_model.generate(pil_image=reference_image,
                                       prompt=prompt,
                                       negative_prompt='worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting',
                                       control_image=[
                                           # inpaint_control_image,
                                           canny_control_image_init,
                                           # canny_control_image_ref,
                                           depth_control_image,
                                           # seg_control_image,
                                           # normal_control_image
                                       ],
                                       image=init_img,
                                       mask_image=mask,
                                       num_samples=1,
                                       num_inference_steps=num_inference_step,
                                       seed=-1,
                                       strength=strength,
                                       scale=scale,
                                       controlnet_conditioning_scale=controlnet_conditioning_scale,
                                       control_guidance_start=control_guidance_start,
                                       control_guidance_end=control_guidance_end,
                                       guidance_scale=guidance,
                                       )
        else:
            images = self.ip_model.generate(pil_image=reference_image,
                                            negative_prompt='worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting',
                                            control_image=[
                                                # inpaint_control_image,
                                                canny_control_image_init,
                                                # canny_control_image_ref,
                                                depth_control_image,
                                                # seg_control_image,
                                                # normal_control_image
                                            ],
                                            image=init_img,
                                            mask_image=mask,
                                            num_samples=1,
                                            num_inference_steps=num_inference_step,
                                            seed=-1,
                                            strength=strength,
                                            scale=scale,
                                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                                            control_guidance_start=control_guidance_start,
                                            control_guidance_end=control_guidance_end,
                                            guidance_scale=guidance,
                                            )
        image = images[0].resize(init_image.size)
        return image


def get_demo(generate_handler):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                # input_image = gr.ImageEditor(
                #     type="pil",
                #     interactive=True,
                #     brush=gr.Brush(colors=[BRUSH_COLOR], color_mode="fixed"),
                # )
                init_image = gr.Image(type="pil", label="Initial Image")
                mask_image = gr.Image(type="pil", label="Mask Image")
                reference_image = gr.Image(type="pil", label="Reference Image")
                object_type = gr.Dropdown(
                    ["Wall", "Floor", "Ceiling", "Furniture", 'Countertop', 'Backsplash'],
                    label="Segmented object type",
                    value="Furniture",
                    interactive=True,
                )

                style_type = gr.Dropdown(
                    ["only_style", "whole_room"],
                    label="Style type",
                    interactive=True,
                )
                brightness = gr.Slider(
                    minimum=0.5,
                    maximum=2,
                    value=0.1,
                    label="Brightness adjustment",
                    interactive=True,
                )

                canny_scale = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=1,
                    label="Canny control scale",
                    interactive=True,
                    step=0.1,
                )

                canny_endpoint = gr.Slider(
                    minimum=0.1,
                    maximum=1,
                    value=1,
                    label="Canny endpoint",
                    interactive=True,
                    step=0.05,
                )

                depth_scale = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=1,
                    label="Depth control scale",
                    interactive=True,
                    step=0.1,
                )

                depth_endpoint = gr.Slider(
                    minimum=0.1,
                    maximum=1,
                    value=1,
                    label="Depth endpoint",
                    interactive=True,
                    step=0.05,
                )

                num_inference_step = gr.Slider(
                    minimum=1,
                    maximum=80,
                    value=32,
                    label="num_inference_step",
                    interactive=True,
                    step=1,
                )

                strength = gr.Slider(
                    minimum=0.1,
                    maximum=1,
                    value=1,
                    label="Strength adjustment",
                    interactive=True,
                    step=0.05,
                )

                scale = gr.Slider(
                    minimum=0.1,
                    maximum=1,
                    value=0.9,
                    label="Scale adjustment",
                    interactive=True,
                    step=0.05,
                )

                guidance = gr.Slider(
                    minimum=1,
                    maximum=25,
                    value=7,
                    label="guidance scale",
                    interactive=True,
                    step=1,
                )
                sampler_type = gr.Dropdown(
                    ["UniPCMultistepScheduler", "DPMSolverMultistepScheduler", "DEISMultistepScheduler"],
                    label="Sampler Type",
                    value="UniPCMultistepScheduler",
                    interactive=True,
                )
                generate_btn = gr.Button("Generate", size="")

            with gr.Column():
                output_image = gr.Image(type="pil", format='jpg', interactive=False)
                # auto_fill_button = gr.Button("Auto Fill Prompt", size="sm")
                # negative_prompt = gr.Textbox(
                #     "Low quality, bad quality, blur, blurry, sketches, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime",
                #     lines=3,
                #     label="Negative Prompt",
                #     interactive=True,
                # )

        generate_btn.click(
            generate_handler,
            [
                object_type,
                init_image,
                mask_image,
                reference_image,
                style_type,
                brightness,
                canny_scale,
                depth_scale,
                canny_endpoint,
                depth_endpoint,
                num_inference_step,
                strength,
                scale,
                guidance,
                sampler_type,
            ],
            output_image,
        )
    return demo

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
    env_vars = load_env_vars()
    generate_handler = Generator()
    demo = get_demo(generate_handler)
    demo.launch(
        server_name="0.0.0.0",
        server_port=8500,
    )
