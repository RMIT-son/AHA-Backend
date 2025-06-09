import time
from typing import Dict, Any, AsyncGenerator
import dspy
from database.schemas import QueryInput, Message
from services.model_manager import model_manager
from modules.text_processing.rag_engine import hybrid_search, rrf
from rich import print

class ResponseHandler:
    """Handles different types of response generation."""
    
    @staticmethod
    def handle_llm_response(prompt: str) -> AsyncGenerator[str, None]:
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
    async def handle_rag_response(input_data: QueryInput, collection_name: str = "dermatology") -> Dict[str, Any]:
        """Handle RAG response with context retrieval."""
        start_time = time.time()
        
        try:
            # Retrieve context using hybrid search
            points = await hybrid_search(
                query=input_data.query, 
                collection_name=collection_name, 
                limit=10
            )
            
            # Apply RRF to get the best context
            context = rrf(points=points, n_points=2)
            
            # Generate response using RAG
            rag_responder = model_manager.get_model("rag_responder")
            response = await rag_responder.forward(context=context, prompt=input_data.query)
            
            execution_time = time.time() - start_time
            print(f"RAG inference took [green]{execution_time:.2f} seconds[/green]")

            return {"response": response}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    async def handle_dynamic_response(input_data: Message) -> Dict[str, Any]:
        """Handle dynamic response with classification-based routing."""
        start_time = time.time()
        
        try:
            # Classify the query to determine response type
            classifier = model_manager.get_model("classifier")
            task_definition = await classifier.forward(prompt=input_data.content)
            
            # Route based on classification
            if task_definition == "non-medical":
                llm_responder = model_manager.get_model("llm_responder")
                response = await llm_responder.forward(prompt=input_data.content)
            else:
                # Use RAG for medical/specialized queries
                points = await hybrid_search(
                    query=input_data.content, 
                    collection_name=task_definition, 
                    limit=10
                )
                context = rrf(points=points, n_points=2)
                
                rag_responder = model_manager.get_model("rag_responder")
                response = await rag_responder.forward(context=context, prompt=input_data.content)
            
            execution_time = time.time() - start_time
            print(f"Dynamic response inference took [green]{execution_time:.2f} seconds[/green]")

            return {
                "task_definition": task_definition,
                "response": response
            }
        except Exception as e:
            raise Exception(f"Dynamic response failed: {str(e)}")