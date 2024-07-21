from base64 import b64decode, b64encode
from io import BytesIO
from typing import Literal

import requests
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from PIL import Image
from pydantic import BaseModel, Field


def image_to_base64(image: Image.Image, format="PNG") -> str:
    buffered = BytesIO()
    image.save(buffered, format=format)
    buffered.seek(0)
    return b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_image(base64_string: str) -> Image.Image:
    decoded = b64decode(base64_string)
    image = Image.open(BytesIO(decoded))
    return image


app = FastAPI(
    title="Image Processing API",
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


class FlipImageRequest(BaseModel):
    image_base64: str = Field(
        ...,
        title="Image Base64",
        description="Base64 encoded image",
    )
    direction: Literal["vertical", "horizontal"] = Field(
        ...,
        title="Direction",
        description="Direction to flip the image",
    )


class FlipImageResponse(BaseModel):
    flipped_image_base64: str = Field(
        ...,
        title="Flipped Image Base64",
        description="Base64 encoded flipped image",
    )


@app.post("/flip-image", response_model=FlipImageResponse)
def flip_image(request: FlipImageRequest):
    if request.direction == "vertical":
        flip = Image.FLIP_TOP_BOTTOM
    else:
        flip = Image.FLIP_LEFT_RIGHT

    image: Image.Image = base64_to_image(request.image_base64)
    flipped_image = image.transpose(flip)
    flipped_image_base64 = image_to_base64(flipped_image)
    return {"flipped_image_base64": flipped_image_base64}


class MyAPIClient:
    def __init__(self, url: str):
        self.url = url

    def flip_image(
        self, image: Image.Image, direction: Literal["vertical", "horizontal"]
    ) -> Image.Image:
        body = {"image_base64": image_to_base64(image), "direction": direction}
        response = requests.post(f"{self.url}/flip-image", json=body)
        response.raise_for_status()
        result = base64_to_image(response.json()["flipped_image_base64"])
        return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=api_host, port=api_port)
