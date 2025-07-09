import time
from rich import print
from app.schemas.message import Message
from typing import AsyncGenerator
from .response_manager import ResponseManager
from app.utils.image_processing import convert_to_dspy_image

class ImageHandler(ResponseManager):
    """Handler specialized for image-only inputs."""

    @classmethod
    async def handle_image_response(cls, input_data: Message = None, user_id: str = None) -> AsyncGenerator[str, None]:
        """
        Handle image-only messages by classifying the image and routing the request accordingly.

        This method first performs classification to determine whether the image is medical-related.
        Based on the classification result, it either routes the input to a RAG (Retrieval-Augmented Generation)
        pipeline or to a general LLM responder.

        Args:
            input_data (Message): The message containing the image to process.

        Yields:
            AsyncGenerator[str, None]: A stream of generated response chunks.

        Raises:
            Exception: If classification or routing fails.
        """
        try:
            # Classify image
            image_result = await cls._classify_image(input_data=input_data)
            print(f"Image classification completed: {image_result}")
            
            # Route based on classification
            return await cls._route_image_response(input_data=input_data, image_result=image_result, user_id=user_id)

        except Exception as e:
            print(f"Image response handling failed: {str(e)}")
            raise Exception(f"Image response failed: {str(e)}")

    @classmethod
    async def _classify_image(cls, input_data: Message = None) -> str:
        """
        Perform classification on the input image to determine its content.

        First attempts general image classification. If the result indicates it's medical-related,
        proceeds with a fine-grained disease classification.

        Args:
            input_data (Message): The message containing the image to classify.

        Returns:
            str: Classification label or disease prediction result.

        Raises:
            Exception: If classification fails.
        """
        start_time = time.time()
        try:
            # Get classifier and classify image
            classifier = await cls.get_classifier()
            image_result = await classifier.classify_image(image=input_data.image)
            cls._log_execution_time(start_time, "Image Classification")
            return image_result
            
        except Exception as e:
            print(f"Image classification failed: {str(e)}")
            raise Exception(f"Image classification failed: {str(e)}")

    @classmethod
    async def _route_image_response(cls, input_data: Message = None, image_result: str = None, user_id: str = None) -> AsyncGenerator[str, None]:
        """
        Route the classified image to the appropriate response handler based on classification result.

        - If the image is medical-related, forwards to RAG responder using disease name as collection.
        - If not, converts image to DSPy format and sends to general LLM.

        Args:
            input_data (Message): The original message object containing image and optional metadata.
            image_result (str): The classification label for the image.

        Yields:
            AsyncGenerator[str, None]: A stream of generated response text.

        Raises:
            Exception: If routing fails or downstream handlers encounter errors.
        """
        try:
            is_medical = image_result != "not-medical-related"

            if is_medical:
                classifier = await cls.get_classifier()
                input_data.image = await classifier.classify_disease(image=input_data.image)
                return await cls.handle_rag_response(input_data=input_data, collection_name=image_result, user_id=user_id)
            else:
                # Non-medical image - use general LLM
                input_data.image = convert_to_dspy_image(image_data=input_data.image)
                return await cls.handle_llm_response(input_data=input_data, user_id=user_id)
                
        except Exception as e:
            print(f"Image response routing failed: {str(e)}")
            raise Exception(f"Image response routing failed: {str(e)}")