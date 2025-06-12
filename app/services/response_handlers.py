import time
import dspy
from database import Message
from typing import Dict, Any, AsyncGenerator
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
    def handle_llm_response(prompt: str = None) -> AsyncGenerator[str, None]:
        """Handle LLM-only response without context, and return async generator."""
        start_time = time.time()

        try:
            llm_responder = model_manager.get_model("llm_responder")
            stream_predict = dspy.streamify(
                llm_responder.response,
                stream_listeners=[dspy.streaming.StreamListener(signature_field_name="response")]
            )
            output_stream = stream_predict(prompt=prompt)
            execution_time = time.time() - start_time
            print(f"LLM inference took {execution_time:.2f} seconds")
            return output_stream
        except Exception as e:
            raise RuntimeError(f"Stream inference error: {str(e)}")

    @staticmethod
    async def handle_rag_response(input_data: Message = None, collection_name: str = "dermatology") -> Dict[str, Any]:
        """Handle RAG response with context retrieval."""
        start_time = time.time()
        
        try:
            # Retrieve context using hybrid search
            points = await hybrid_search(
                query=input_data.content, 
                collection_name=collection_name, 
                limit=10
            )
            
            # Apply RRF to get the best context
            context = rrf(points=points, n_points=2)
            
            # Generate response using RAG
            rag_responder = model_manager.get_model("rag_responder")
            response = await rag_responder.forward(context=context, prompt=input_data.content)
            
            execution_time = time.time() - start_time
            print(f"RAG inference took [green]{execution_time:.2f} seconds[/green]")

            return {"response": response}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    async def handle_dynamic_response(input_data: Message = None) -> AsyncGenerator[str, None]:
        """Handle dynamic response with classification-based routing."""
        start_time = time.time()
        
        try:
            prompt = await translate_text(text=input_data.content, dest="en")
            # Classify the query to determine response type
            classifier = model_manager.get_model("classifier")
            task_definition = classifier.forward(prompt=prompt.text)
            print(task_definition)
            execution_time = time.time() - start_time
            print(f"Classify inference took {execution_time:.2f} seconds")

            # Route based on classification
            if task_definition == "non-medical":
                llm_responder = model_manager.get_model("llm_responder")
                stream_predict = dspy.streamify(
                    llm_responder.response,
                    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="response")]
                )
                output_stream = stream_predict(prompt=input_data.content)

            else:
                # Use RAG for medical/specialized queries
                points = await hybrid_search(
                    query=prompt.text, 
                    collection_name=task_definition, 
                    limit=10
                )
                # context = await translate_text(text=rrf(points=points, n_points=2), src=prompt.dest, dest=prompt.src)
                context = rrf(points=points, n_points=2)

                rag_responder = model_manager.get_model("rag_responder")
                stream_predict = dspy.streamify(
                    rag_responder.response,
                    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="response")]
                )
                output_stream = stream_predict(context=context, prompt=input_data.content)
            
            execution_time = time.time() - start_time
            print(f"Dynamic response inference took [green]{execution_time:.2f} seconds[/green]")

            return output_stream
        except Exception as e:
            raise Exception(f"Dynamic response failed: {str(e)}")