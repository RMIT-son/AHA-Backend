import time
import asyncio
from rich import print
from database import Message
from typing import AsyncGenerator
from ..manage_responses import TextHandler, ImageHandler
from app.modules.image_processing import convert_to_dspy_image

class TextImageHandler(TextHandler, ImageHandler):
    """Handler specialized for text+image inputs."""

    @classmethod
    async def handle_text_image_response(cls, input_data: Message = None, user_id: str = None) -> AsyncGenerator[str, None]:
        """Handle text+image response with parallel classification and routing."""
        start_time = time.time()
        try:
            # Run text and image classification in parallel
            text_task = cls._classify_text(input_data)
            image_task = cls._classify_image(input_data)
            
            text_result, image_result = await asyncio.gather(text_task, image_task)
            
            print(f"Text classification: {text_result}, Image classification: {image_result}")
            cls._log_execution_time(start_time, "Text+Image Classification")
            
            # Route based on classification
            return await cls._route_text_image_response(
                input_data=input_data, 
                text_result=text_result, 
                image_result=image_result,
                user_id=user_id
            )

        except Exception as e:
            print(f"Text+Image response handling failed: {str(e)}")
            raise Exception(f"Text+Image response failed: {str(e)}")


    @classmethod
    async def _route_text_image_response(cls, input_data: Message = None, text_result: str = None, image_result: str = None, user_id: str = None) -> AsyncGenerator[str, None]:
        """Route text+image response based on classification."""
        try:
            text_is_medical = text_result != "not-medical-related"
            image_is_medical = image_result != "not-medical-related"
            
            if text_is_medical or image_is_medical:
                # At least one is medical - use RAG
                # Prioritize text classification for collection name
                collection_name = text_result if text_is_medical else image_result
                
                if image_is_medical:
                    # Process medical image
                    input_data.image = image_result
                else:
                    # Convert non-medical image
                    input_data.image = convert_to_dspy_image(image_data=input_data.image)
                
                return await cls.handle_rag_response(input_data=input_data, collection_name=collection_name, user_id=user_id)
            else:
                # Both non-medical - use general LLM
                input_data.image = convert_to_dspy_image(image_data=input_data.image)
                return await cls.handle_llm_response(input_data=input_data, user_id=user_id)
                
        except Exception as e:
            print(f"Text+Image response routing failed: {str(e)}")
            raise Exception(f"Text+Image response routing failed: {str(e)}")