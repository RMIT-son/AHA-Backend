import time
import dspy
from rich import print
from database import Message
from typing import Any, AsyncGenerator, Awaitable
from ..manage_models.model_manager import model_manager
from app.modules.image_processing import convert_to_dspy_image
from app.modules.text_processing import (
    hybrid_search,
    rrf,
)

class ResponseManager:
    """Base handler for different types of response generation."""

    @classmethod
    def _log_execution_time(cls, start_time: float = None, process_name: str = None) -> None:
        """Helper method to log execution time."""
        execution_time = time.time() - start_time
        color = "[green]" if "RAG" in process_name or "Dynamic" in process_name else ""
        end_color = "[/green]" if color else ""
        print(f"{process_name} inference took {color}{execution_time:.2f} seconds{end_color}")
    
    @classmethod
    def _create_stream_predict(cls, model: dspy.Module = None, signature_field_name: str = "response") -> Awaitable[Any]:
        """Helper method to create streamified prediction."""
        return dspy.streamify(
            model.response,
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name=signature_field_name)]
        )

    @classmethod
    def handle_llm_response(cls, input_data: Message = None) -> AsyncGenerator[str, None]:
        """Handle LLM-only response without context, and return async generator."""
        start_time = time.time()

        try:
            llm_responder = model_manager.get_model("llm_responder")
            stream_predict = cls._create_stream_predict(llm_responder)
            output_stream = stream_predict(prompt=input_data.content, image=input_data.image)
            cls._log_execution_time(start_time, "LLM")
            return output_stream
        except Exception as e:
            raise RuntimeError(f"Stream inference error: {str(e)}")

    @classmethod
    async def handle_rag_response(cls, input_data: Message = None, collection_name: str = None) -> AsyncGenerator[str, None]:
        """Handle RAG response with context retrieval."""
        start_time = time.time()

        try:
            compose_text_image_prompt = f"Prompt: {input_data.content}\n\n{input_data.image}" if input_data.image else input_data.content
            # Retrieve context using hybrid search
            points = await hybrid_search(
                query=compose_text_image_prompt, 
                collection_name=collection_name, 
                limit=4
            )
            
            # Apply RRF to get the best context
            context = rrf(points=points, n_points=2)
            
            # Generate response using RAG
            rag_responder = model_manager.get_model("rag_responder")
            stream_predict = cls._create_stream_predict(rag_responder)
            output_stream = stream_predict(context=context, prompt=input_data.content, image=input_data.image)

            cls._log_execution_time(start_time, "RAG")

            return output_stream
        except Exception as e:
            raise Exception(f"RAG response failed: {str(e)}")

    @classmethod
    async def summarize(cls, input_data: Message = None) -> str:
        """Summarize conversation based on user prompt"""
        try:
            summarizer = model_manager.get_model("summarizer")

            image_data = input_data.image
            if isinstance(image_data, (bytes, bytearray)):
                image = convert_to_dspy_image(image_data=image_data)
            elif isinstance(image_data, str) and image_data.lower() not in ["", "string", "none"]:
                image = convert_to_dspy_image(image_data=image_data)
            else:
                image = None

            summarized_context = await summarizer.forward(image=image, prompt=input_data.content)
            return summarized_context

        except Exception as e:
            raise Exception(f"Summarized failed: {str(e)}")
        
    @classmethod
    async def get_classifier(cls) -> dspy.Module:
        """Get classifier model."""
        try:
            classifier = model_manager.get_model("classifier")
            return classifier
        except Exception as e:
            print(f"Failed to load classifier: {str(e)}")
            raise Exception(f"Classifier loading failed: {str(e)}")