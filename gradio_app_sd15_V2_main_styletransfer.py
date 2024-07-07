from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler, AutoencoderKL, UniPCMultistepScheduler
from gradio_imageslider import ImageSlider
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from PIL import Image
from transformers import pipeline
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from ada_pallete import ada_palette
from controlnet_aux import NormalBaeDetector, MLSDdetector
from ip_adapter import IPAdapter, IPAdapterPlus, IPAdapterPlusXL
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, StableDiffusionControlNetPipeline
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
from dependencies import ImageCaptioningClient, SegmentationClient
from utils import expand_mask, hash_image, image_to_base64, shrink_mask
from ada_pallete import ada_palette

SEGMENT_COLOR = [216, 98, 188, 255]
BRUSH_COLOR = "rgb(216, 98, 188)"

resolution = {
    'tiny': 256,
    'low': 512,
    'medium': 768,
    'high': 1024
}

torch.backends.cuda.matmul.allow_tf32 = True


def resize_image_with_height(pil_image, new_height):
    # Calculate the aspect ratio
    width, height = pil_image.size
    aspect_ratio = width / height

    # Calculate the new width based on the aspect ratio
    new_width = int(new_height * aspect_ratio) 
    new_width = new_width - new_width % 8
    # Resize the image while preserving the aspect ratio
    resized_image = pil_image.resize((new_width, new_height))

    # Return the resized image
    return resized_image


class SegmentHandler:
    def __init__(self, segmentation_client):
        self.segmentation_client = segmentation_client
        self.__data = {
            "image_hash": None,
            "service_type": None,
            "segmentation_map": None,
            "id_to_label": None,
        }

    def __get_segmentation_data(
        self,
        image: Image.Image,
        service_type: Literal["Wall", "Floor", "Ceiling", "Furniture"],
    ):
        image_hash = hash_image(image)
        if (self.__data["image_hash"] == image_hash) and (
            self.__data["service_type"] == service_type
        ):
            return self.__data["segmentation_map"], self.__data["id_to_label"]
        else:
            segmentations = self.segmentation_client(image, service_type=service_type)
            image_width, image_height = image.size
            segmentation_map = np.zeros((image_height, image_width), dtype=np.int64)
            id_to_label = {}
            for d in segmentations:
                id_ = len(id_to_label) + 1
                id_to_label[id_] = d["label"]
                segmentation_map[d["mask"]] = id_

            self.__data["image_hash"] = image_hash
            self.__data["service_type"] = service_type
            self.__data["segmentation_map"] = segmentation_map
            self.__data["id_to_label"] = id_to_label
            return segmentation_map, id_to_label


    def new_method(self, image, service_type):
        segmentations = self.segmentation_client(image, service_type=service_type)
        return segmentations

    def __call__(
        self,
        image_edit_data: dict,
        service_type: Literal["Wall", "Floor", "Ceiling", "Furniture"],
    ):
        image = image_edit_data["background"]
        layer_1 = image_edit_data["layers"][0]
        layer_1_mask = np.asarray(layer_1)[..., 3] != 0
        segmentation_map, id_to_label = self.__get_segmentation_data(
            image, service_type
        )
        layer_1_ids = np.unique(segmentation_map[layer_1_mask])
        # remove 0
        layer_1_ids = layer_1_ids[layer_1_ids != 0]
        mask = np.isin(segmentation_map, layer_1_ids) | layer_1_mask
        # if mask_expand > 0:
        #     mask = expand_mask(mask, mask_expand)
        # if mask_shrink > 0:
        #     mask = shrink_mask(mask, mask_shrink)
        mask = np.dstack([mask] * 4)

        new_layer_1 = mask * SEGMENT_COLOR
        new_layer_1 = Image.fromarray(new_layer_1.astype(np.uint8))
        new_composite = Image.alpha_composite(image, new_layer_1)
        return {
            "layers": [new_layer_1],
            "background": image,
            "composite": new_composite,
        }


class Generator:
    def __init__(self):
        
        base_model_path = "runwayml/stable-diffusion-v1-5"
        vae_model_path = "vae_model/vae-ft-mse-840000-ema-pruned.safetensors"
        device = "cuda"
        image_encoder_path = "models/image_encoder/"
        # ip_ckpt = "models/ip-adapter-plus_sd15.bin"
        ip_ckpt = "models/ip-adapter_sd15.bin"
        # models/ip-adapter_sd15.bin
        # device = "cuda"
        torch.cuda.empty_cache()

        # vae = AutoencoderKL.from_single_file(vae_model_path, torch_dtype=torch.float16).to("cuda")
        
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        self.mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
        self.depth_estimator = pipeline('depth-estimation')
        
        controlnet_mlsd = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
        )

        # checkpoint_inpaint = "lllyasviel/control_v11p_sd15_inpaint"
        # controlnet_inpaint = ControlNetModel.from_pretrained(
        #     checkpoint_inpaint, torch_dtype=torch.float16
        # )
        checkpoint_canny = "lllyasviel/control_v11p_sd15_canny"
        controlnet_canny = ControlNetModel.from_pretrained(checkpoint_canny, torch_dtype=torch.float16)
        
        checkpoint_depth = "lllyasviel/control_v11f1p_sd15_depth"
        controlnet_depth = ControlNetModel.from_pretrained(checkpoint_depth, torch_dtype=torch.float16)
        
        checkpoint_seg = "lllyasviel/control_v11p_sd15_seg"
        controlnet_seg = ControlNetModel.from_pretrained(checkpoint_seg, torch_dtype=torch.float16)
        
        checkpoint_normal = "lllyasviel/control_v11p_sd15_normalbae"
        controlnet_normal = ControlNetModel.from_pretrained(checkpoint_normal, torch_dtype=torch.float16)
        
        controlnet = [controlnet_mlsd, controlnet_canny, controlnet_depth, controlnet_seg, controlnet_normal]

        self.image_processor_b = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        self.image_segmentor_b = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

        self.normal_processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
        
        # noise_scheduler = DDIMScheduler(
        # num_train_timesteps=1000,
        # beta_start=0.00085,
        # beta_end=0.012,
        # beta_schedule="scaled_linear",
        # clip_sample=False,
        # set_alpha_to_one=False,
        # steps_offset=1,)
        # vae = AutoencoderKL.from_single_file(vae_model_path).to(dtype=torch.float16)
        # self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
        # base_model_path,
        # controlnet=controlnet,
        # torch_dtype=torch.float16,
        # scheduler=noise_scheduler,
        # vae=vae, 
        # ).to(device)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "Lykon/dreamshaper-8-inpainting",  controlnet=controlnet, torch_dtype=torch.float16,
        # vae=vae
        )
        self.pipe.to(device)

        # self.pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
        # "/home/kasra/realisticVisionV60B1_v60B1InpaintingVAE.safetensors",  controlnet=controlnet, torch_dtype=torch.float16,
        # # vae=vae
        # ).to(device)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.unet = register_cross_attention_hook(self.pipe.unet)
        # self.ip_model = IPAdapterPlus(self.pipe, image_encoder_path, ip_ckpt, device="cuda", num_tokens=16)
        self.ip_model = IPAdapter(self.pipe, image_encoder_path, ip_ckpt, device="cuda")

    def canny_image(self, image):
        
        image1 = np.array(image)
        low_threshold = 100
        high_threshold = 200
        image1 = cv2.Canny(image1, low_threshold, high_threshold)
        image1 = image1[:, :, None]
        image1 = np.concatenate([image1, image1, image1], axis=2)
        control_image = Image.fromarray(image1).resize(image.size)
        
        return control_image


    def normal_image(self, image: Image.Image):
        
        processor = self.normal_processor
        control_image = processor(image)
        np_image = np.array(control_image)
        control_image = Image.fromarray(np_image).resize(image.size)
        
        return control_image


    def depth_image(self, image: Image.Image):
        
        depth_image = self.depth_estimator(image)['depth']
        np_image = np.array(depth_image) 
        np_image = np_image[:, :, None]
        np_image = np.concatenate([np_image, np_image, np_image], axis=2)
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


    def make_inpaint_condition(self, image, image_mask):
        """
        Prepares an image for inpainting by converting the masked areas to grayscale,
        preserving the shadow and variation details from the original image.

        Args:
        image (PIL.Image): The input image, in RGB.
        image_mask (PIL.Image): The mask image, in grayscale (single channel).

        Returns:
        PIL.Image: The modified image with masked areas in grayscale.
        """
        # Convert images to arrays
        image_array = np.array(image.convert("RGB"), dtype=np.float32)
        image_mask_array = np.array(image_mask.convert("L"), dtype=np.float32) / 255.0

        # Ensure the images are the same size
        assert image_array.shape[:2] == image_mask_array.shape[:2], "Image and image_mask must have the same dimensions."

        # Create a grayscale version of the image
        grayscale_array = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # grayscale_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
        grayscale_array = grayscale_array[:, :, None]
        grayscale_array = np.concatenate([grayscale_array]*3, axis=2)
        # Replace only the masked areas with grayscale
        mask_condition = image_mask_array > 0.5  # Mask where values are greater than 0.5
        image_array[mask_condition] = grayscale_array[mask_condition]

        # Normalize and convert to uint8 for PIL compatibility
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        pil_image = Image.fromarray(image_array, 'RGB')

        return pil_image


    def seg_image(self, image):
        image_processor_b = self.image_processor_b
        image_segmentor_b = self.image_segmentor_b
        pixel_values = image_processor_b(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = image_segmentor_b(pixel_values)
        seg = image_processor_b.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
        for label, color in enumerate(ada_palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        control_image = Image.fromarray(color_seg)
        return control_image   

    def crop_centre(self, image: Image.Image):

        im = image
        width, height = im.size   # Get dimensions
        new_width = width - 15
        new_height = height -15
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        # Crop the center of the image
        cropped_im = im.crop((left, top, right, bottom))
        final_im = resize_image_with_height(cropped_im, 256)
        return final_im

    def __call__(
            self,
            object_type,
            init_image,
            # mask_image,
            reference_image,
            style_type,
            brightness,
            mlsd_scale,
            canny_scale,
            depth_scale,
            seg_scale,
            normal_scale,
            mlsd_endpoint,
            canny_endpoint,
            depth_endpoint,
            seg_endpoint,
            normal_endpoint,
            num_inference_step,
            strength,
            scale,
            guidance,
            sampler_type,
            resolution_mode,
            ):
        
        init_image['background'] = resize_image_with_height(init_image['background'], resolution[resolution_mode]).convert('RGB')
        main_image =  init_image['background']
        reference_image = self.crop_centre(reference_image)
        mask_image = init_image['layers'][0].convert('L').convert('RGB').resize(init_image['background'].size, Image.Resampling.LANCZOS)
        mask_image = np.array(mask_image)
        mask_image[mask_image > 0]=255
        mask_image = Image.fromarray(mask_image)
        init_image['background'] = init_image['background'].convert('RGB')
        mask_target_img = ImageChops.lighter(init_image['background'], mask_image)
        invert_target_mask = ImageChops.invert(mask_image)
        gray_target_image = init_image['background'].convert('L').convert('RGB')
        gray_target_image = ImageEnhance.Brightness(gray_target_image)
        gray_target_image = gray_target_image.enhance(brightness)
        grayscale_img = ImageChops.darker(gray_target_image, mask_image)
        img_black_mask = ImageChops.darker(init_image['background'], invert_target_mask)
        grayscale_init_img = ImageChops.lighter(img_black_mask, grayscale_img)
        init_image = grayscale_init_img
        init_img = init_image
        mask = mask_image.resize(main_image.size, Image.Resampling.LANCZOS)
        mask = self.pipe.mask_processor.blur(mask, blur_factor=3)
        
        if sampler_type == "UniPCMultistepScheduler":
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_type == "DPMSolverMultistepScheduler":
            self.pipe.scheduler.config['use_karras_sigmas'] = True
            self.pipe.scheduler.config['algorithm_type'] = "sde-dpmsolver++"
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_type == "DEISMultistepScheduler":
            self.pipe.scheduler = DEISMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        prompt = self.Generate_caption(reference_image)
        prompt = object_type + ' covered by' + prompt.split('of')[-1]

        controlnet_conditioning_scale = [mlsd_scale,
                                        canny_scale,
                                        depth_scale,
                                        seg_scale,
                                        normal_scale,]
        control_guidance_start = [0, 0, 0, 0, 0]
        control_guidance_end = [mlsd_endpoint,
                                canny_endpoint,
                                depth_endpoint,
                                seg_endpoint,
                                normal_endpoint,]
        # inpaint_control_image = self.make_inpaint_condition(main_image, mask_image)
        canny_control_image_init = self.canny_image(main_image)
        depth_control_image = self.depth_image(main_image)
        normal_control_image = self.normal_image(main_image)
        seg_control_image = self.seg_image(main_image)
        mlsd_control_image = self.mlsd(main_image).resize(main_image.size)

        with torch.inference_mode():
            if style_type == 'only_style':
                images = self.ip_model.generate(pil_image=reference_image,
                                        prompt=prompt,
                                        negative_prompt='worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting',
                                        control_image=[
                                            mlsd_control_image,
                                            canny_control_image_init,
                                            # canny_control_image_ref,
                                            depth_control_image,
                                            seg_control_image,
                                            normal_control_image
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
                image_1 = self.pipe.image_processor.apply_overlay(mask, init_image, images[0])
            else:
                images = self.ip_model.generate(pil_image=reference_image,
                                                negative_prompt='worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting',
                                                control_image=[
                                                    mlsd_control_image,
                                                    canny_control_image_init,
                                                    # canny_control_image_ref,
                                                    depth_control_image,
                                                    seg_control_image,
                                                    normal_control_image
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
                image_1 = self.pipe.image_processor.apply_overlay(mask, init_image, images[0])
        image = images[0].resize(init_image.size)
        return (image, image_1)


def get_demo(generate_handler, segment_handler):
    with gr.Blocks() as demo:
        gr.HTML("<h1 style='text-align: center;'>Regional Style Transfer Demo</h1>",)
        with gr.Row():
            # gr.Markdown("1. **Import Images**: Upload your main image and the style image.\t 2. **Brush Selection**: Use the brush tool to highlight areas on the main image where the style will be applied.\t3. **Segment**: Click the **Segment** button to preview the selected areas.\4. **Adjust Segmentation**: Modify the segmentation by adding or removing parts as needed.\t5. **Set Parameters**: Adjust the parameters for the style transfer process.\t6. **Generate Image**: Hit the **Generate** button and wait for about 30 seconds to see your stylized image.")
            with gr.Column():
                init_image = gr.ImageEditor(
                    type="pil",
                    interactive=True,
                    brush=gr.Brush(colors=[BRUSH_COLOR], color_mode="fixed"),
                )
                # init_image = gr.Image(type="pil", label="Initial Image")
                # mask_image = gr.Image(type="pil", label="Mask Image")
                reference_image = gr.Image(type="pil", label="Reference Image")
                object_type = gr.Dropdown(
                    ["Wall", "Floor", "Ceiling", "Furniture", 'Countertop', 'Backsplash'],
                    label="Segmented object type",
                    value="Floor",
                    interactive=True,
                )
                # mask_expand = gr.Slider(
                #     minimum=0,
                #     maximum=10,
                #     value=0,
                #     label="Mask Expansion",
                #     interactive=True,
                #     step=1,
                # )

                # mask_shrink = gr.Slider(
                #     minimum=0,
                #     maximum=10,
                #     value=0,
                #     label="Mask Shrinkage",
                #     interactive=True,
                #     step=1,
                # )
                segment_btn = gr.Button("Segment")

                style_type = gr.Dropdown(
                    ["only_style", "whole_room"],
                    label="Style type",
                    interactive=True,
                )
                
            with gr.Column():
                img_slider = ImageSlider(label="Blur image", type="pil", slider_color="pink")
                # img_slider.upload(fn, inputs=img1, outputs=img1)
                # output_image = gr.Image(type="pil", format='jpg', interactive=False)
                
                # auto_fill_button = gr.Button("Auto Fill Prompt", size="sm")
                # negative_prompt = gr.Textbox(
                #     "Low quality, bad quality, blur, blurry, sketches, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime",
                #     lines=3,
                #     label="Negative Prompt",
                #     interactive=True,
                # )
                brightness = gr.Slider(
                    minimum=0.5,
                    maximum=2,
                    value=1,
                    label="Brightness adjustment",
                    interactive=True,
                )

                mlsd_scale = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=0.5,
                    label="mlsd control scale",
                    interactive=True,
                    step=0.1,
                )

                mlsd_endpoint = gr.Slider(
                    minimum=0.1,
                    maximum=1,
                    value=0.6,
                    label="mlsd endpoint",
                    interactive=True,
                    step=0.05,
                )


                canny_scale = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=0,
                    label="Canny control scale",
                    interactive=True,
                    step=0.1,
                )

                canny_endpoint = gr.Slider(
                    minimum=0.1,
                    maximum=1,
                    value=0.1,
                    label="Canny endpoint",
                    interactive=True,
                    step=0.05,
                )

                depth_scale = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=0.6,
                    label="Depth control scale",
                    interactive=True,
                    step=0.1,
                )

                depth_endpoint = gr.Slider(
                    minimum=0.1,
                    maximum=1,
                    value=0.6,
                    label="Depth endpoint",
                    interactive=True,
                    step=0.05,
                )

                
                seg_scale = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=0,
                    label="Segmentation control scale",
                    interactive=True,
                    step=0.1,
                )

                seg_endpoint = gr.Slider(
                    minimum=0.1,
                    maximum=1,
                    value=0.1,
                    label="Segmentation endpoint",
                    interactive=True,
                    step=0.05,
                )
                 
                
                normal_scale = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=0.8,
                    label="Normal control scale",
                    interactive=True,
                    step=0.1,
                )

                normal_endpoint = gr.Slider(
                    minimum=0.1,
                    maximum=1,
                    value=0.75,
                    label="Normal endpoint",
                    interactive=True,
                    step=0.05,
                )
                 

                num_inference_step = gr.Slider(
                    minimum=1,
                    maximum=80,
                    value=40,
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
                    value=0.8,
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

                resolution_mode = gr.Radio(
                    ["tiny", "low", "medium", "high"],
                    label="Resolution Mode",
                    value="medium",
                )
                generate_btn = gr.Button("Generate", size="")




        segment_btn.click(
            segment_handler,
            [init_image, object_type],
            init_image,
        )
        generate_btn.click(
            generate_handler,
            [
                object_type,
                init_image,
                # mask_image,
                reference_image,
                style_type,
                brightness,
                mlsd_scale,
                canny_scale,
                depth_scale,
                seg_scale,
                normal_scale,
                mlsd_endpoint,
                canny_endpoint,
                depth_endpoint,
                seg_endpoint,
                normal_endpoint,
                num_inference_step,
                strength,
                scale,
                guidance,
                sampler_type,
                resolution_mode,
            ],
            img_slider,
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
    segmentation_client = SegmentationClient(
        env_vars["segmentation_api_url"],
        env_vars["temp_bucket_name"],
        env_vars["env"],
    )

    image_captioning_client = ImageCaptioningClient(
        image_captioning_api_url=env_vars["image_captioning_api_url"],
        temp_bucket_name=env_vars["temp_bucket_name"],
        env=env_vars["env"],
    )
    generate_handler = Generator()
    segment_handler = SegmentHandler(segmentation_client)
    demo = get_demo(generate_handler, segment_handler)
    demo.launch(
        server_name="0.0.0.0",
        server_port=8400,
    )
