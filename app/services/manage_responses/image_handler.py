import time
from rich import print
from database import Message
from typing import AsyncGenerator
from .response_manager import ResponseManager
from app.modules.image_processing import convert_to_dspy_image
from database.gcs_client import upload_file_to_gcs

class ImageHandler(ResponseManager):
    """Handler specialized for image-only inputs."""

    @classmethod
    async def handle_image_response(cls, input_data: Message = None) -> AsyncGenerator[str, None]:
        """Handle image-only response with classification and routing."""
        start_time = time.time()
        try:
            # Classify image
            image_result = await cls._classify_image(input_data=input_data)
            print(f"Image classification completed: {image_result}")
            
            cls._log_execution_time(start_time, "Image Classification")
            
            # Route based on classification
            return await cls._route_image_response(input_data=input_data, image_result=image_result)

        except Exception as e:
            print(f"Image response handling failed: {str(e)}")
            raise Exception(f"Image response failed: {str(e)}")

    @classmethod
    async def _classify_image(cls, input_data: Message = None) -> str:
        """Classify image input."""
        try:
            # Get classifier and classify image
            classifier = await cls.get_classifier()
            # If image is raw bytes, upload to GCS and replace with URL
            if isinstance(input_data.image, bytes):
                gcs_url = await upload_file_to_gcs(file_bytes=input_data.image)
                input_data.image = gcs_url
            image_result = await classifier.classify_image(image=input_data.image)
            if image_result != "not-medical-related":
                image_result = await classifier.classify_disease(image=input_data.image)
            return image_result
            
        except Exception as e:
            print(f"Image classification failed: {str(e)}")
            raise Exception(f"Image classification failed: {str(e)}")

    @classmethod
    async def _route_image_response(cls, input_data: Message = None, image_result: str = None) -> AsyncGenerator[str, None]:
        """Route image response based on classification with parallel processing."""
        try:
            is_medical = image_result != "not-medical-related"

            if is_medical:
                input_data.image = image_result
                return await cls.handle_rag_response(input_data=input_data, collection_name=image_result)
            else:
                # Non-medical image - use general LLM
                input_data.image = convert_to_dspy_image(image_data=input_data.image)
                return cls.handle_llm_response(input_data=input_data)
                
        except Exception as e:
            print(f"Image response routing failed: {str(e)}")
            raise Exception(f"Image response routing failed: {str(e)}")