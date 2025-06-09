from contextlib import asynccontextmanager
from fastapi import APIRouter
from database.schemas import QueryInput
from services.model_manager import model_manager
from services.response_handlers import ResponseHandler

@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager for model loading and cleanup."""
    try:
        # Load and warm up models on startup
        model_manager.load_models()
        await model_manager.warmup_models()
        print("Application startup completed successfully!")
        
        yield
        
    except Exception as e:
        print(f"Error during startup: {e}")
        raise
    finally:
        # Clean up models on shutdown
        model_manager.cleanup_models()
        print("Application shutdown completed successfully!")


# Create router
router = APIRouter(prefix="/api/response", tags=["Modules"])


@router.post("/llm")
def llm_response(input_data: QueryInput):
    """
    Generate response using LLM module directly without any context or classification.
    
    Args:
        input_data: Query input containing the user's question
        
    Returns:
        Dictionary containing the generated response or error message
    """
    return ResponseHandler.handle_llm_response(input_data)


@router.post("/rag")
async def rag_response(input_data: QueryInput):
    """
    Generate response using RAG (Retrieval-Augmented Generation).
    Retrieves relevant context from Qdrant and uses it to generate response.
    
    Args:
        input_data: Query input containing the user's question
        
    Returns:
        Dictionary containing the generated response or error message
    """
    return await ResponseHandler.handle_rag_response(input_data)


@router.post("/dynamic-response")
async def dynamic_response(input_data: QueryInput):
    """
    Generate response using dynamic routing based on query classification.
    First classifies the input query, then chooses between LLM-only or RAG
    depending on the task type (e.g., 'medical' vs 'non-medical').
    
    Args:
        input_data: Query input containing the user's question
        
    Returns:
        Dictionary containing task definition and generated response
    """
    return await ResponseHandler.handle_dynamic_response(input_data)