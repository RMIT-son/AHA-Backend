import asyncio
from rich import print
from typing import AsyncGenerator
from app.schemas.message import Message
from app.utils.common import classify_file
from .response_manager import ResponseManager
from app.utils.image_processing import convert_to_dspy_image

class ImageHandler(ResponseManager):
    """Handler specialized for image-only inputs."""

    @classmethod
    async def handle_image_response(cls, input_data: Message = None, user_id: str = None) -> AsyncGenerator[str, None]:
        """
        Convert image data from base64 to dspy image for LLM

        Args:
            input_data (Message): The message containing the image to process.

        Yields:
            AsyncGenerator[str, None]: A stream of generated response chunks.

        Raises:
            Exception: If classification or routing fails.
        """
        try:
            # Extract image files (e.g., image/jpeg, image/png, etc.)
            image_files, _ = classify_file(input_data.files or [])
            # Convert each image URL to dspy format
            dspy_images = await asyncio.gather(
                *(convert_to_dspy_image(f.url) for f in image_files)
            )
            input_data.files[0].url = dspy_images[0]
            return await cls.handle_llm_response(input_data=input_data, user_id=user_id)
        except Exception as e:
            print(f"Image response handling failed: {str(e)}")
            raise Exception(f"Image response failed: {str(e)}")