import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
from ip_adapter import IPAdapter

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "models/image_encoder/"
ip_ckpt = "models/ip-adapter_sd15.bin"
device = "cuda"


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

# read image prompt
image = Image.open("/home/kasra/Downloads/3 (1)scaled2.jpg/")
image.resize((512, 512))

masked_image = Image.open("/home/kasra/PycharmProjects/reimagine-segmentation/data/image_after_resize/image_after_resize.png/").resize((512, 768))
mask = Image.open("/home/kasra/PycharmProjects/reimagine-segmentation/data/image_after_resize/_floor14.jpg/").resize((512, 768))
image_grid([masked_image.resize((512, 512)), mask.resize((512, 512))], 1, 2)

# load ip-adapter
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
# generate
images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50,
                           seed=42, image=masked_image, mask_image=mask, strength=0.7, guidance_scale=13, )
grid = image_grid(images, 1, 1)
grid.show()
