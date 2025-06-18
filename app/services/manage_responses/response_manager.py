import time
import dspy
from rich import print
from database import Message
from typing import Any, AsyncGenerator, Awaitable
from ..manage_models.model_manager import model_manager
from app.modules.text_processing import (
    hybrid_search,
    rrf,
)

class ResponseManager:
    """Base handler for different types of response generation."""

    @staticmethod
    def _log_execution_time(start_time: float = None, process_name: str = None) -> None:
        """Helper method to log execution time."""
        execution_time = time.time() - start_time
        color = "[green]" if "RAG" in process_name or "Dynamic" in process_name else ""
        end_color = "[/green]" if color else ""
        print(f"{process_name} inference took {color}{execution_time:.2f} seconds{end_color}")
    
    @staticmethod
    def _create_stream_predict(model: dspy.Module = None, signature_field_name: str = "response") -> Awaitable[Any]:
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
            stream_predict = ResponseManager._create_stream_predict(llm_responder)
            output_stream = stream_predict(prompt=input_data.content, image=input_data.image)
            ResponseManager._log_execution_time(start_time, "LLM")
            return output_stream
        except Exception as e:
            raise RuntimeError(f"Stream inference error: {str(e)}")

    @staticmethod
    async def handle_rag_response(input_data: Message = None, collection_name: str = None) -> AsyncGenerator[str, None]:
        """Handle RAG response with context retrieval."""
        start_time = time.time()

        try:
            compose_text_image_prompt = f"Prompt: {input_data.content}\n\n{input_data.image}" if input_data.image else input_data.content
            # Retrieve context using hybrid search
            points = await hybrid_search(
                query=compose_text_image_prompt, 
                collection_name=collection_name, 
                limit=10
            )
            
            # Apply RRF to get the best context
            context = rrf(points=points, n_points=3)
            
            # Generate response using RAG
            rag_responder = model_manager.get_model("rag_responder")
            stream_predict = ResponseManager._create_stream_predict(rag_responder)
            output_stream = stream_predict(context=context, prompt=input_data.content, image=input_data.image)

            ResponseManager._log_execution_time(start_time, "RAG")

            return output_stream
        except Exception as e:
            raise Exception(f"RAG response failed: {str(e)}")

    @classmethod
    async def get_classifier(cls) -> dspy.Module:
        """Get classifier model."""
        try:
            classifier = model_manager.get_model("classifier")
            return classifier
        except Exception as e:
            print(f"Failed to load classifier: {str(e)}")
            raise Exception(f"Classifier loading failed: {str(e)}")

    @classmethod
    async def handle_dynamic_response(cls, input_data: Message = None) -> AsyncGenerator[str, None]:
        """Handle dynamic response with optimized classification-based routing."""
        try:
            
            # Input validation
            if not input_data:
                print(f"No input data provided")
                raise Exception("No input data provided")
            
            # Get classifier once
            classifier = await cls.get_classifier()
            
            # Determine input types
            has_text = bool(input_data.content and input_data.content.strip())
            has_image = input_data.image is not None
            
            
            # Import handlers here to avoid circular imports
            from .text_handler import TextHandler
            from .image_handler import ImageHandler
            from .text_image_handler import TextImageHandler
            
            # Route to appropriate handler based on input type
            if has_text and has_image:
                # Both inputs - use TextImageHandler
                text_image_handler = TextImageHandler()
                return await text_image_handler.handle_text_image_response(input_data=input_data, classifier=classifier)
            elif has_text:
                # Text only - use TextHandler
                text_handler = TextHandler()
                return await text_handler.handle_text_response(input_data=input_data, classifier=classifier)
            elif has_image:
                # Image only - use ImageHandler
                image_handler = ImageHandler()
                return await image_handler.handle_image_response(input_data=input_data, classifier=classifier)
            else:
                raise Exception("No valid input provided (neither text nor image)")
            
        except Exception as e:
            print(f"handle_dynamic_response failed: {str(e)}")
            raise Exception(f"Dynamic response failed: {str(e)}")