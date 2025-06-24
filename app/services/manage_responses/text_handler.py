import time
import asyncio
from rich import print
from database import Message
from typing import AsyncGenerator
from ..translator import translate_text
from .response_manager import ResponseManager

class TextHandler(ResponseManager):
    """Handler specialized for text-only inputs."""
    
    @classmethod
    async def handle_text_response(cls, input_data: Message = None) -> AsyncGenerator[str, None]:
        """Handle text-only response with classification and routing."""        
        try:
            # Classify text
            text_result = await cls._classify_text(input_data)
            print(f"Text classification completed: {text_result}")
            
            
            # Route based on classification
            return await cls._route_text_response(input_data, text_result)
            
        except Exception as e:
            print(f"Text response handling failed: {str(e)}")
            raise Exception(f"Text response failed: {str(e)}")
    
    @classmethod
    async def _classify_text(cls, input_data: Message = None) -> str:
        """Classify text input with parallel processing."""
        try:
            # Run translation and classifier loading in parallel
            translate_task = translate_text(text=input_data.content, dest="en")
            classifier_task = cls.get_classifier()
            
            start_time = time.time()

            translated_prompt, classifier = await asyncio.gather(translate_task, classifier_task)
            # Classify text
            text_result = await classifier.classify_text(prompt=translated_prompt.text)
            
            cls._log_execution_time(start_time, "Text Classification")
            return text_result
            
        except Exception as e:
            print(f"Text classification failed: {str(e)}")
            raise Exception(f"Text classification failed: {str(e)}")
    
    @classmethod
    async def _route_text_response(cls, input_data: Message = None, text_result: str = None) -> AsyncGenerator[str, None]:
        """Route text response based on classification."""
        try:
            is_medical = text_result != "not-medical-related"
            
            if is_medical:
                # Medical text - use RAG
                return await cls.handle_rag_response(input_data=input_data, collection_name=text_result)
            else:
                # Non-medical text - use general LLM
                return cls.handle_llm_response(input_data=input_data)
                
        except Exception as e:
            print(f"Text response routing failed: {str(e)}")
            raise Exception(f"Text response routing failed: {str(e)}")