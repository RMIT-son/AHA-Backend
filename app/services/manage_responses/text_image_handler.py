import dspy
import time
import asyncio
from rich import print
from database import Message
from typing import AsyncGenerator
from .text_handler import TextHandler
from .image_handler import ImageHandler
from .response_manager import ResponseManager
from app.modules.image_processing import convert_to_dspy_image


class TextImageHandler(ResponseManager):
    """Handler specialized for inputs containing both text and images."""
    
    def __init__(self):
        """Initialize with text and image handlers."""
        self.text_handler = TextHandler()
        self.image_handler = ImageHandler()
    
    async def handle_text_image_response(self, input_data: Message = None, classifier: dspy.Module = None) -> AsyncGenerator[str, None]:
        """Handle response for inputs containing both text and images."""
        start_time = time.time()
        
        try:
            
            # Validate that both text and image are present
            has_text = bool(input_data.content and input_data.content.strip())
            has_image = input_data.image is not None
            
            if not (has_text and has_image):
                raise Exception("TextImageHandler requires both text and image inputs")
            
            # Classify both text and image in parallel using the respective handlers
            text_result, image_result = await self._classify_both_inputs(input_data=input_data, classifier=classifier)

            print(f"Text: {text_result}, Image: {image_result}")
            self._log_execution_time(start_time, "Text-Image Classification")
            
            # Route based on combined classification results
            return await self._route_combined_response(input_data=input_data, text_result=text_result, image_result=image_result, classifier=classifier)

        except Exception as e:
            print(f"Text-image response handling failed: {str(e)}")
            raise Exception(f"Text-image response failed: {str(e)}")
    
    async def _classify_both_inputs(self, input_data: Message = None, classifier: dspy.Module = None) -> tuple[str, str]:
        """Classify both text and image inputs using their respective handlers."""
        try:
            
            # Use the individual handlers' classification methods in parallel
            text_result, image_result = await asyncio.gather(
                self.text_handler._classify_text(input_data=input_data, classifier=classifier),
                self.image_handler._classify_image(input_data=input_data, classifier=classifier)
            )
            
            return text_result, image_result
            
        except Exception as e:
            print(f"Combined classification failed: {str(e)}")
            raise Exception(f"Combined classification failed: {str(e)}")

    async def _route_combined_response(self, input_data: Message = None, text_result: str = None, image_result: str = None, classifier: dspy.Module = None) -> AsyncGenerator[str, None]:
        """Route response based on combined text and image classification results."""
        try:
            is_text_medical = text_result != "not-medical-related"
            is_image_medical = image_result != "not-medical-related"
            
            # Priority routing logic
            if is_text_medical and is_image_medical:
                # Both are medical - prioritize text classification but include image context
                return await self._handle_dual_medical_response(input_data=input_data, text_result=text_result, image_result=image_result, classifier=classifier)

            elif is_text_medical and not is_image_medical:
                # Only text is medical - use text-based RAG
                input_data.image = convert_to_dspy_image(image_data=input_data.image)
                return await self.text_handler._route_text_response(input_data=input_data, text_result=text_result)

            elif is_image_medical and not is_text_medical:
                # Only image is medical - use image-based RAG
                return await self.image_handler._route_image_response(input_data=input_data, image_result=image_result, classifier=classifier)

            else:
                # Neither is medical - use general LLM with both inputs
                input_data.image = convert_to_dspy_image(image_data=input_data.image)
                return self.handle_llm_response(input_data=input_data)
                
        except Exception as e:
            print(f"Combined response routing failed: {str(e)}")
            raise Exception(f"Combined response routing failed: {str(e)}")
    
    async def _handle_dual_medical_response(self, input_data: Message = None, text_result: str = None, image_result: str = None, classifier: dspy.Module = None) -> AsyncGenerator[str, None]:
        """Handle case where both text and image are medical-related."""
        try:
            print(f"Text: {text_result}, Image: {image_result}")
            
            # If both classifications point to the same medical domain, use that
            if text_result == image_result:
                print(f"Same medical domain detected: {text_result}")
                # Enhance image with disease classification for better context
                input_data.image = await classifier.classify_disease(image=input_data.image)

            else:
                # Different medical domains - prioritize text but enhance with image context
                print(f"Different medical domains - prioritizing text: {text_result}")

            
            return await self.handle_rag_response(input_data=input_data, collection_name=text_result)
            
        except Exception as e:
            print(f"Dual medical response handling failed: {str(e)}")
            raise Exception(f"Dual medical response failed: {str(e)}")