from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def Generate_caption(image_path: str):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    raw_image = Image.open(image_path).convert('RGB')

    # unconditional image captioning
    text = "image of"
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)
