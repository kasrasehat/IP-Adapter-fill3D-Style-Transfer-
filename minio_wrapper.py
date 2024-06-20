"""Wrapper for Minio client to download and upload images in PIL format."""

from io import BytesIO

from minio import Minio
from PIL import Image


class MinioWrapper:
    """Wrapper for Minio client to download and upload images in PIL format.

    Args:
        endpoint (str): Minio endpoint.
        access_key (str): Minio access key.
        secret_key (str): Minio secret key.
        secure (bool): Use secure connection.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool,
    ):
        self.minio = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    def download_image(self, bucket_name: str, object_name: str) -> Image.Image:
        """Download an image from Minio.

        Args:
            bucket_name (str): Bucket name.
            object_name (str): Object name.

        Returns:
            Image.Image: PIL image.
        """
        response = self.minio.get_object(
            bucket_name=bucket_name, object_name=object_name
        )
        buffer = BytesIO(response.read())
        image = Image.open(buffer)
        return image

    def upload_image(
        self, image: Image.Image, bucket_name: str, object_name: str
    ) -> None:
        """Upload an image to Minio.

        Args:
            image (Image.Image): PIL image.
            bucket_name (str): Bucket name.
            object_name (str): Object name.

        Returns:
            None
        """
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        self.minio.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=buffer,
            length=buffer.getbuffer().nbytes,
        )
