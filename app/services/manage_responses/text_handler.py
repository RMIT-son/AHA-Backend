from rich import print
from typing import AsyncGenerator
from app.schemas.message import Message
from .response_manager import ResponseManager

class TextHandler(ResponseManager):
    """Handler specialized for text-only inputs."""
    
    @classmethod
    async def handle_text_response(cls, input_data: Message = None) -> AsyncGenerator[str, None]:
        """
        Handle a user message that contains only text

        Args:
            input_data (Message): The user message containing the text input.
            user_id (str): The user's ID, used for retrieving past messages and history.

        Yields:
            AsyncGenerator[str, None]: A stream of generated response tokens from the selected model.

        Raises:
            Exception: If routing fails.
        """        
        try:
            if input_data.context:
                return await cls.handle_rag_response(input_data=input_data)
            else:
                return await cls.handle_llm_response(input_data=input_data)
        except Exception as e:
            print(f"Text response handling failed: {str(e)}")
            raise Exception(f"Text response failed: {str(e)}")