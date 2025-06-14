import time
import dspy
import asyncio
from database import Message, Optional
from typing import AsyncGenerator
from .model_manager import model_manager
from app.modules import (
    hybrid_search, 
    rrf
)
from rich import print
from .translate import translate_text

class ResponseHandler:
    """Handles different types of response generation."""
    
    @staticmethod
    def _create_stream_predict(model, signature_field_name="response"):
        """Helper method to create streamified prediction."""
        return dspy.streamify(
            model.response,
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name=signature_field_name)]
        )
    
    @staticmethod
    def _log_execution_time(start_time: float, process_name: str):
        """Helper method to log execution time."""
        execution_time = time.time() - start_time
        color = "[green]" if "RAG" in process_name or "Dynamic" in process_name else ""
        end_color = "[/green]" if color else ""
        print(f"{process_name} inference took {color}{execution_time:.2f} seconds{end_color}")
    
    @staticmethod
    def handle_llm_response(input_data: Message = None, image: Optional[str | dspy.Image] = None) -> AsyncGenerator[str, None]:
        """Handle LLM-only response without context, and return async generator."""
        start_time = time.time()

        try:
            llm_responder = model_manager.get_model("llm_responder")
            stream_predict = ResponseHandler._create_stream_predict(llm_responder)
            output_stream = stream_predict(prompt=input_data.content, image=image)
            ResponseHandler._log_execution_time(start_time, "LLM")
            return output_stream
        except Exception as e:
            raise RuntimeError(f"Stream inference error: {str(e)}")

    @staticmethod
    async def handle_rag_response(input_data: Message = None, collection_name: str = None, image: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Handle RAG response with context retrieval."""
        start_time = time.time()
        
        try:
            # Retrieve context using hybrid search
            points = await hybrid_search(
                query=input_data.content, 
                collection_name=collection_name, 
                limit=15
            )
            
            # Apply RRF to get the best context
            context = rrf(points=points, n_points=2)

            # Generate response using RAG
            rag_responder = model_manager.get_model("rag_responder")
            stream_predict = ResponseHandler._create_stream_predict(rag_responder)
            output_stream = stream_predict(context=context, prompt=input_data.content, image=image)
            
            ResponseHandler._log_execution_time(start_time, "RAG")

            return output_stream
        except Exception as e:
            raise Exception(f"RAG response failed: {str(e)}")
    
    @staticmethod
    async def handle_dynamic_response(input_data: Message = None) -> AsyncGenerator[str, None]:
        """Handle dynamic response with classification-based routing."""
        start_time = time.time()
        
        try:
            # Translate user's prompt into English
            translated_prompt = await translate_text(text=input_data.content, dest="en")

            # Classify the query to determine response type
            classifier = model_manager.get_model("classifier")
            text_classification = asyncio.create_task(classifier.classify_text(prompt=translated_prompt.text))
            image_classification = asyncio.create_task(classifier.classify_image(image=input_data.image))
            text_result, image_result = await asyncio.gather(text_classification, image_classification)
            print(f"text: {text_result}")
            print(f"image: {image_result}")
            ResponseHandler._log_execution_time(start_time, "Classify")

            # Route based on classification
            if text_result != "not-medical-related" and image_result != "not-medical-related":
                disease_classification = await classifier.classify_disease(image=input_data.image)
                print(f"disease: {disease_classification}")
                return await ResponseHandler.handle_rag_response(input_data=input_data, collection_name=text_result, image=disease_classification)
            elif text_result != "not-medical-related" and image_result == "not-medical-related":
                return await ResponseHandler.handle_rag_response(input_data=input_data, collection_name=text_result)
            else:
                return ResponseHandler.handle_llm_response(input_data=input_data, image=input_data.image)

        except Exception as e:
            raise Exception(f"Dynamic response failed: {str(e)}")