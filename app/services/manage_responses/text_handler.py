import dspy
import time
from rich import print
from database import Message
from typing import AsyncGenerator
from ..translator import translate_text
from .response_manager import ResponseManager

class TextHandler(ResponseManager):
    """Handler specialized for text-only inputs."""
    
    async def handle_text_response(self, input_data: Message = None, classifier: dspy.Module = None) -> AsyncGenerator[str, None]:
        """Handle text-only response with classification and routing."""
        start_time = time.time()
        
        try:
            
            # Classify text
            text_result = await self._classify_text(input_data, classifier)
            print(f"Text classification completed: {text_result}")
            
            self._log_execution_time(start_time, "Text Classification")
            
            # Route based on classification
            return await self._route_text_response(input_data, text_result)
            
        except Exception as e:
            print(f"Text response handling failed: {str(e)}")
            raise Exception(f"Text response failed: {str(e)}")
    
    async def _classify_text(self, input_data: Message = None, classifier: dspy.Module = None) -> str:
        """Classify text input."""
        try:
            # Translate text
            translated_prompt = await translate_text(text=input_data.content, dest="en")
            
            # Classify text
            text_result = await classifier.classify_text(prompt=translated_prompt.text)
            return text_result
            
        except Exception as e:
            print(f"Text classification failed: {str(e)}")
            raise Exception(f"Text classification failed: {str(e)}")
    
    async def _route_text_response(self, input_data: Message = None, text_result: str = None) -> AsyncGenerator[str, None]:
        """Route text response based on classification."""
        try:
            is_medical = text_result != "not-medical-related"
            
            if is_medical:
                # Medical text - use RAG
                return await self.handle_rag_response(input_data=input_data, collection_name=text_result)
            else:
                # Non-medical text - use general LLM
                return self.handle_llm_response(input_data=input_data)
                
        except Exception as e:
            print(f"Text response routing failed: {str(e)}")
            raise Exception(f"Text response routing failed: {str(e)}")