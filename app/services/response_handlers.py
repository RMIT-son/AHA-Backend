import time
import dspy
import asyncio
from database import Message
from typing import AsyncGenerator
from .model_manager import model_manager
from app.modules.image_processing import (
    read_image_from_url,
    read_image_from_local
)
from app.modules.text_processing import (
    hybrid_search,
    rrf,
)
from rich import print
from .translate import translate_text

class ResponseHandler:
    """Handles different types of response generation."""

    @staticmethod
    def _log_execution_time(start_time: float = None, process_name: str = None):
        """Helper method to log execution time."""
        execution_time = time.time() - start_time
        color = "[green]" if "RAG" in process_name or "Dynamic" in process_name else ""
        end_color = "[/green]" if color else ""
        print(f"{process_name} inference took {color}{execution_time:.2f} seconds{end_color}")
    
    @staticmethod
    def _create_stream_predict(model: dspy.Module = None, signature_field_name="response"):
        """Helper method to create streamified prediction."""
        return dspy.streamify(
            model.response,
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name=signature_field_name)]
        )
    
    @staticmethod
    def handle_llm_response(input_data: Message = None) -> AsyncGenerator[str, None]:
        """Handle LLM-only response without context, and return async generator."""
        start_time = time.time()

        try:
            llm_responder = model_manager.get_model("llm_responder")
            stream_predict = ResponseHandler._create_stream_predict(llm_responder)
            output_stream = stream_predict(prompt=input_data.content, image=input_data.image)
            ResponseHandler._log_execution_time(start_time, "LLM")
            return output_stream
        except Exception as e:
            raise RuntimeError(f"Stream inference error: {str(e)}")

    @staticmethod
    async def handle_rag_response(input_data: Message = None, collection_name: str = None) -> AsyncGenerator[str, None]:
        """Handle RAG response with context retrieval."""
        start_time = time.time()

        try:
            compose_text_image_prompt = f"Prompt: {input_data.content} {input_data.image}" if input_data.image else input_data.content
            # Retrieve context using hybrid search
            points = await hybrid_search(
                query=compose_text_image_prompt, 
                collection_name=collection_name, 
                limit=15
            )
            
            # Apply RRF to get the best context
            context = rrf(points=points, n_points=3)

            # Generate response using RAG
            rag_responder = model_manager.get_model("rag_responder")
            stream_predict = ResponseHandler._create_stream_predict(rag_responder)
            output_stream = stream_predict(context=context, prompt=input_data.content, image=input_data.image)

            ResponseHandler._log_execution_time(start_time, "RAG")

            return output_stream
        except Exception as e:
            raise Exception(f"RAG response failed: {str(e)}")
    
    
    @staticmethod
    async def handle_dynamic_response(input_data: Message = None) -> AsyncGenerator[str, None]:
        """Handle dynamic response with classification-based routing."""
        start_time = time.time()
        
        try:
            # Translate user's prompt to English
            translated_prompt = await translate_text(text=input_data.content, dest="en")
            
            # Classify text and image
            classifier = model_manager.get_model("classifier")
            
            if input_data.image:
                # Run both classifications in parallel when image exists
                text_result, image_result = await asyncio.gather(
                    classifier.classify_text(prompt=translated_prompt.text),
                    classifier.classify_image(image=input_data.image)
                )
            else:
                # Only classify text when no image
                text_result = await classifier.classify_text(prompt=translated_prompt.text)
                image_result = None
            
            print(f"Classifications - Text: {text_result}, Image: {image_result}")
            ResponseHandler._log_execution_time(start_time, "Classify")
            
            # Determine response routing based on classification
            is_text_medical = text_result != "not-medical-related"
            is_image_medical = image_result and image_result != "not-medical-related"
            
            if is_image_medical:
                # Medical image detected - classify disease and use RAG
                input_data.image = await classifier.classify_disease(image=input_data.image)
                return await ResponseHandler.handle_rag_response(
                    input_data=input_data, 
                    collection_name=image_result
                )
            
            elif is_text_medical:
                # Only text is medical-related - use RAG with processed image
                input_data.image = read_image_from_url(input_data.image) if input_data.image else None
                return await ResponseHandler.handle_rag_response(
                    input_data=input_data, 
                    collection_name=text_result
                )
            
            else:
                # Neither text nor image is medical - use LLM only
                input_data.image = read_image_from_url(input_data.image) if input_data.image else None
                return ResponseHandler.handle_llm_response(input_data=input_data)
                
        except Exception as e:
            raise Exception(f"Dynamic response failed: {str(e)}")