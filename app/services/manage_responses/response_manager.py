import time
import dspy
import asyncio
from rich import print
from app.schemas.message import Message
from typing import Any, AsyncGenerator, Awaitable
from ..manage_models.model_manager import model_manager
from app.utils.image_processing import convert_to_dspy_image

class ResponseManager:
    """Base handler for different types of response generation."""

    @classmethod
    def _log_execution_time(cls, start_time: float = None, process_name: str = None) -> None:
        """
        Log the time taken to execute a process.

        Args:
            start_time (float): The timestamp when the process started.
            process_name (str): A descriptive name of the process (e.g., "LLM", "RAG").

        Returns:
            None
        """
        execution_time = time.time() - start_time
        color = "[green]" if "RAG" in process_name or "Dynamic" in process_name else ""
        end_color = "[/green]" if color else ""
        print(f"{process_name} inference took {color}{execution_time:.2f} seconds{end_color}")

    @classmethod
    def _create_stream_predict(cls, model: dspy.Module = None, signature_field_name: str = "response") -> Awaitable[Any]:
        """
        Wrap a DSPy module to enable streaming predictions.

        Args:
            model (dspy.Module): The DSPy model to be used for streaming.
            signature_field_name (str): Field name used to identify the streaming signature.

        Returns:
            Awaitable[Any]: A callable stream prediction function.
        """
        return dspy.streamify(
            model.response,
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name=signature_field_name)]
        )

    @classmethod
    async def handle_llm_response(cls, input_data: Message) -> AsyncGenerator[str, None]:
        """
        Handle a user request using an LLM-only response (no external knowledge/context).

        Args:
            input_data (Message): The user message containing prompt and optionally image.
            user_id (str): ID of the user, used for retrieving past message context.

        Yields:
            AsyncGenerator[str, None]: A stream of generated response text.

        Raises:
            RuntimeError: If inference fails or model access fails.
        """
        start_time = time.time()
        try:
            llm_responder = model_manager.get_model("llm_responder")
            stream_predict = cls._create_stream_predict(llm_responder)
            output_stream = stream_predict(
                prompt=input_data.content, 
                images=input_data.images, 
                recent_conversations=input_data.recent_conversations
            )
            cls._log_execution_time(start_time, "LLM")
            return output_stream
        except Exception as e:
            raise RuntimeError(f"Stream inference error: {str(e)}")

    @classmethod
    async def handle_rag_response(cls, input_data: Message) -> AsyncGenerator[str, None]:
        """
        Handle a user request using RAG (Retrieval-Augmented Generation) with context retrieval.

        Args:
            input_data (Message): The user message including content and optional image.

        Yields:
            AsyncGenerator[str, None]: A stream of generated response text.

        Raises:
            Exception: If retrieval or generation fails.
        """
        start_time = time.time()
        try:
            rag_responder = model_manager.get_model("rag_responder")
            stream_predict = cls._create_stream_predict(rag_responder)
            output_stream = stream_predict(
                context=input_data.context, 
                prompt=input_data.content, 
                images=input_data.images, 
                recent_conversations=input_data.recent_conversations
            )
            cls._log_execution_time(start_time, "RAG")
            return output_stream
        except Exception as e:
            raise Exception(f"RAG response failed: {str(e)}")

    @classmethod
    async def summarize(cls, input_data: Message = None) -> str:
        """
        Summarize the user's prompt and/or image to generate a suitable conversation title.

        Args:
            input_data (Message): The message that contains the prompt and optionally an image.

        Returns:
            str: The generated summary or title.

        Raises:
            Exception: If summarization fails.
        """
        try:
            llm_responder = model_manager.get_model("llm_responder")
            summarizer = model_manager.get_model("summarizer")
            
            if input_data.images and not input_data.content:
                images = await asyncio.gather(*[
                    convert_to_dspy_image(image) for image in input_data.images
                ])
                response = await llm_responder.forward(images=images)
                summarized_context = await summarizer.forward(input=response)
            else:
                prompt = input_data.content
                summarized_context = await summarizer.forward(input=prompt)

            return summarized_context

        except Exception as e:
            raise Exception(f"Summarized failed: {str(e)}")