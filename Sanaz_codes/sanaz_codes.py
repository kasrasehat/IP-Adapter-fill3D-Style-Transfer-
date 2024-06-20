image = [image_depth, segmentation_cond_image, normal_map_image]
        generated_image = pipe(
            prompt=pos_prompt,
            negative_prompt=self.neg_prompt,
            num_inference_steps=num_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=None,
            image=image,
            # mask_image=mask_image,
            ip_adapter_image=ip_image,
            # control_image=[image_depth, segmentation_cond_image],
            controlnet_conditioning_scale=[1.0, 0.6, 0.6],
            controlnet_guidance_end=[0.8, 0.9, 0.5]
        ).images[0]
2:34
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    # "digiplay/Photon_v1",
    # "runwayml/stable-diffusion-v1-5",
    "Lykon/AbsoluteReality",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=dtype
)pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                     weight_name="ip-adapter-plus_sd15.bin")
pipe.set_ip_adapter_scale(0.7)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)