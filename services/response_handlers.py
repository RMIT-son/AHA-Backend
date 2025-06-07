import time
from typing import Dict, Any
from database.schemas import QueryInput
from services.model_manager import model_manager
from modules.text_processing.rag_engine import hybrid_search, rrf
from rich import print


class ResponseHandler:
    """Handles different types of response generation."""
    
    @staticmethod
    def handle_llm_response(input_data: QueryInput) -> Dict[str, Any]:
        """Handle LLM-only response without context."""
        start_time = time.time()
        
        try:
            llm_responder = model_manager.get_model("llm_responder")
            response = llm_responder.forward(prompt=input_data.query)
            
            execution_time = time.time() - start_time
            print(f"LLM inference took [green]{execution_time:.2f} seconds[/green]")
            # print(model_manager.get_history())
            
            return {"response": response}
        except Exception as e:
            return {"error": str(e)}
    
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
            response = rag_responder.forward(context=context, prompt=input_data.query)
            
            execution_time = time.time() - start_time
            print(f"RAG inference took [green]{execution_time:.2f} seconds[/green]")
            # print(model_manager.get_history())

            return {"response": response}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    async def handle_dynamic_response(input_data: QueryInput) -> Dict[str, Any]:
        """Handle dynamic response with classification-based routing."""
        start_time = time.time()
        
        try:
            # Classify the query to determine response type
            classifier = model_manager.get_model("classifier")
            task_definition = classifier.forward(prompt=input_data.query)
            
            # Route based on classification
            if task_definition == "non-medical":
                llm_responder = model_manager.get_model("llm_responder")
                response = llm_responder.forward(prompt=input_data.query)
            else:
                # Use RAG for medical/specialized queries
                points = await hybrid_search(
                    query=input_data.query, 
                    collection_name=task_definition, 
                    limit=10
                )
                context = rrf(points=points, n_points=2)
                
                rag_responder = model_manager.get_model("rag_responder")
                response = rag_responder.forward(context=context, prompt=input_data.query)
            
            execution_time = time.time() - start_time
            print(f"Dynamic response inference took [green]{execution_time:.2f} seconds[/green]")
            # print(model_manager.get_history())

            return {
                "task_definition": task_definition,
                "response": response
            }
        except Exception as e:
            raise Exception(f"Dynamic response failed: {str(e)}")