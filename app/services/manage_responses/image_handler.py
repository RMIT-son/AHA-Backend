import time
import dspy
from rich import print
from database import Message
from typing import AsyncGenerator
from .response_manager import ResponseManager
from app.modules.image_processing import convert_to_dspy_image

class ImageHandler(ResponseManager):
    """Handler specialized for image-only inputs."""
    
    async def handle_image_response(self, input_data: Message = None, classifier: dspy.Module = None) -> AsyncGenerator[str, None]:
        """Handle image-only response with classification and routing."""
        start_time = time.time()
        
        try:            
            # Classify image
            image_result = await self._classify_image(input_data, classifier)
            print(f"Image classification completed: {image_result}")
            
            self._log_execution_time(start_time, "Image Classification")
            
            # Route based on classification
            return await self._route_image_response(input_data, image_result, classifier)
            
        except Exception as e:
            print(f"Image response handling failed: {str(e)}")
            raise Exception(f"Image response failed: {str(e)}")
    
    async def _classify_image(self, input_data: Message = None, classifier: dspy.Module = None) -> str:
        """Classify image input."""
        try:
            # Classify image
            image_result = await classifier.classify_image(image=input_data.image)
            return image_result
            
        except Exception as e:
            print(f"Image classification failed: {str(e)}")
            raise Exception(f"Image classification failed: {str(e)}")

    async def _route_image_response(self, input_data: Message = None, image_result: str = None, classifier: dspy.Module = None) -> AsyncGenerator[str, None]:
        """Route image response based on classification."""
        try:
            is_medical = image_result != "not-medical-related"
            
            if is_medical:
                # Medical image - classify disease and use RAG
                input_data.image = await classifier.classify_disease(image=input_data.image)
                return await self.handle_rag_response(input_data=input_data, collection_name=image_result)
            else:
                # Non-medical image - use general LLM
                input_data.image = convert_to_dspy_image(image_data=input_data.image)
                return self.handle_llm_response(input_data=input_data)
                
        except Exception as e:
            print(f"Image response routing failed: {str(e)}")
            raise Exception(f"Image response routing failed: {str(e)}")