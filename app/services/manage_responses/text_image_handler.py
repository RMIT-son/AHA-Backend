import asyncio
from rich import print
from app.schemas.message import Message
from typing import AsyncGenerator
from ..manage_responses import TextHandler, ImageHandler
from app.utils.image_processing import convert_to_dspy_image

class TextImageHandler(TextHandler, ImageHandler):
    """Handler specialized for text+image inputs."""

    @classmethod
    async def handle_text_image_response(cls, input_data: Message = None, user_id: str = None) -> AsyncGenerator[str, None]:
        """
        Handle a user message that contains both text and image by classifying each in parallel and routing the response.

        Args:
            input_data (Message): The user's message including text and image.
            user_id (str): The user ID used for retrieving history and relevant context.

        Yields:
            AsyncGenerator[str, None]: A stream of tokens generated by the selected model (RAG or LLM).

        Raises:
            Exception: If classification or routing fails.
        """
        try:
            input_data.image = await asyncio.create_task(convert_to_dspy_image(input_data.image))

            # Route based on classification
            return await cls.handle_text_response(input_data=input_data, user_id=user_id)

        except Exception as e:
            print(f"Text+Image response handling failed: {str(e)}")
            raise Exception(f"Text+Image response failed: {str(e)}")