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
from Image_caption import Generate_caption
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

base_model_path = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
# diffusers/stable-diffusion-xl-1.0-inpainting-0.1
vae_model_path = "vae_model/vae-ft-mse-840000-ema-pruned.safetensors"
# vae_model/vae-ft-mse-840000-ema-pruned.safetensors
# stabilityai/sd-vae-ft-mse
image_encoder_path = "models/image_encoder/"
# ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
ip_ckpt = "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
# models/ip-adapter-plus_sd15.bin
# models/ip-adapter_sd15.bin
device = "cuda"
torch.cuda.empty_cache()

init_image = Image.open("/home/kasra/style_transfer/rooms/id6/id6.jpg/")

generator = torch.Generator(device="cuda").manual_seed(1)
mask_image = Image.open("/home/kasra/style_transfer/rooms/id6/_wall25.jpg/").convert('L').convert('RGB').resize(init_image.size, Image.Resampling.LANCZOS)
mask_target_img = ImageChops.lighter(init_image, mask_image)
invert_target_mask = ImageChops.invert(mask_image)
gray_target_image = init_image.convert('L').convert('RGB')
gray_target_image = ImageEnhance.Brightness(gray_target_image)

# Adjust brightness
# The factor 1.0 means original brightness, greater than 1.0 makes the image brighter. Adjust this if the image is too dim
factor = 1.3  # Try adjusting this to get the desired brightness
gray_target_image = gray_target_image.enhance(factor)
grayscale_img = ImageChops.darker(gray_target_image, mask_image)
img_black_mask = ImageChops.darker(init_image, invert_target_mask)
grayscale_init_img = ImageChops.lighter(img_black_mask, grayscale_img)
init_image = grayscale_init_img
init_img = init_image
mask = mask_image.resize(init_image.size)
ref_image_path = "/home/kasra/style_transfer/styles/images (45).jpeg/"
reference_image = Image.open(ref_image_path)
# .resize(init_image.size, Image.Resampling.LANCZOS)

# vae = AutoencoderKL.from_single_file(vae_model_path).to(dtype=torch.float16)
# vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
prompt = Generate_caption(ref_image_path)
prompt = 'floor completely covered by' + prompt.split('of')[-1]
print(prompt)
# prompt = ''


def seg_image(image):
    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
      outputs = image_segmentor(pixel_values)
    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in enumerate(ada_palette):
        color_seg[seg == label, :] = color
    color_seg = color_seg.astype(np.uint8)
    control_image = Image.fromarray(color_seg)
    return control_image


def canny_image(image):
    image = np.array(image)

    low_threshold = 50
    high_threshold = 150

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)
    return control_image


def depth_image(image: Image.Image):
    depth_estimator = pipeline('depth-estimation')
    image = depth_estimator(image)['depth']
    np_image = (np.array(image) / 256).astype('uint8')
    depth_map = Image.fromarray(np_image).resize(init_image.size)
    return depth_map


def normal_image(image: Image.Image):
    processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
    control_image = processor(image)
    return control_image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


# def make_inpaint_condition(image, image_mask):
#     image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
#     image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
#     assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
#     image[image_mask > 0.5] = -1.0  # set as masked pixel
#     image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
#     control_image = torch.from_numpy(image)
#     return control_image
def make_inpaint_condition(image, image_mask):
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
    grayscale_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
    grayscale_array = np.stack([grayscale_array]*3, axis=-1)  # Replicate the grayscale across all color channels

    # Replace only the masked areas with grayscale
    mask_condition = image_mask_array > 0.5  # Mask where values are greater than 0.5
    image_array[mask_condition] = grayscale_array[mask_condition]

    # Normalize and convert to uint8 for PIL compatibility
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    # Convert back to PIL Image
    pil_image = Image.fromarray(image_array, 'RGB')

    return pil_image


# checkpoint_inpaint = "lllyasviel/control_v11p_sd15_inpaint"
# controlnet_inpaint = ControlNetModel.from_pretrained(
#     checkpoint_inpaint, torch_dtype=torch.float16
# )
# checkpoint_canny = "lllyasviel/control_v11p_sd15_canny"
controlnet_canny = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
# )
checkpoint_depth = "diffusers/controlnet-depth-sdxl-1.0"
controlnet_depth = ControlNetModel.from_pretrained(checkpoint_depth, variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
# checkpoint_seg = "lllyasviel/control_v11p_sd15_seg"
# controlnet_seg = ControlNetModel.from_pretrained(checkpoint_seg, torch_dtype=torch.float16)
# checkpoint_normal = "lllyasviel/control_v11p_sd15_normalbae"
# controlnet_normal = ControlNetModel.from_pretrained(checkpoint_normal, torch_dtype=torch.float16)
# controlnet = [controlnet_inpaint, controlnet_canny, controlnet_canny, controlnet_depth, controlnet_seg, controlnet_normal]
controlnet = [controlnet_canny, controlnet_depth]
controlnet_conditioning_scale = [0.7, 1.0]
control_guidance_start = [0, 0]
control_guidance_end = [0.4, 0.8]
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    use_safetensors=True,
    torch_dtype=torch.float16,
    add_watermarker=False,
).to(device)
pipe.unet = register_cross_attention_hook(pipe.unet)
# Uminosachi/realisticVisionV51_v51VAE-inpainting
# Pigatron/sd15_inpaint_Realistic_Vision_V4.0
# runwayml/stable-diffusion-inpainting
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# noise_scheduler = DDIMScheduler(
#     num_train_timesteps=1000,
#     beta_start=0.00085,
#     beta_end=0.012,
#     beta_schedule="scaled_linear",
#     clip_sample=False,
#     set_alpha_to_one=False,
#     steps_offset=1,
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()
# generate image
# normal_control_image = normal_image(init_image)
canny_control_image_init = canny_image(init_image)
# canny_control_image_ref = canny_image(reference_image)
depth_control_image = depth_image(init_image)
# seg_control_image = seg_image(init_image)
# inpaint_control_image = make_inpaint_condition(init_image, mask_image)

# image = pipe(
#     prompt=["floor, black white and gold marble "],
#     image=[init_image],
#     mask_image=mask_image,
#     control_image=[
#         inpaint_control_image,
#         canny_control_image,
#         depth_control_image,
#         seg_control_image,
#         normal_control_image
#     ],
#     vae=vae,
#     num_inference_steps=32,
#     controlnet_conditioning_scale=controlnet_conditioning_scale,
#     control_guidance_start=control_guidance_start,
#     control_guidance_end=control_guidance_end,
#     guidance_scale=13,
# )
# load ip-adapter
# ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device="cuda", num_tokens=16)
# ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device="cuda")
ip_model = IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device="cuda", num_tokens=16)
# generate image variations
images = ip_model.generate(pil_image=reference_image,
                           prompt=prompt,
                           negative_prompt= 'worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting',
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
                            num_inference_steps=32,
                            seed=-1,
                            strength=1,
                            scale=0.9,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            control_guidance_start=control_guidance_start,
                            control_guidance_end=control_guidance_end,
                            guidance_scale=7,
                           )
images[0].resize((1536, 1024)).show()


# images = self.ip_model.generate(pil_image=reference_image, image=[normal_condition, seg_condition, depth_condition, canny_condition], num_samples=num_samples,
#                                    num_inference_steps=num_inference_steps, seed=seed, control_guidance_end=control_guidance_end,
#                                    scale=reference_scale, controlnet_conditioning_scale=condition_scales)

