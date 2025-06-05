import time
from fastapi import APIRouter
from database.schemas import QueryInput
from modules.orchestration.llm_gateway import set_lm_configure
from modules.text_processing.classifier import classify_task
from modules.text_processing.rag_engine import (
    hybrid_search,
    rrf,
    rag_response,
    llm_response,
    llm_config
)

router = APIRouter(prefix="/api/response", tags=["Text"])

set_lm_configure(config=llm_config)

@router.post("/llm")
def llm(input: QueryInput):
    """
    Uses the LLM module directly without any context or classification.
    """
    try:
        start = time.time()
        response = llm_response(prompt=input.query)
        print("LLM inference took", time.time() - start)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@router.post("/rag")
def rag(input: QueryInput):
    """
    Retrieves relevant context from Qdrant and uses RAG to respond.
    """
    try:
        start = time.time()
        points = hybrid_search(query=input.query, collection_name="dermatology", limit=5)
        context = rrf(points=points, n_points=3)
        response = rag_response(context=context, prompt=input.query)
        print("RAG inference took", time.time() - start)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@router.post("/dynamic-response")
def dynamic_response(input: QueryInput):
    """
    First classifies the input query, then dynamically chooses between
    LLM-only or RAG depending on task type (e.g., 'medical').
    """
    try:
        start = time.time()
        task_definition = classify_task(input.query)
        
        if task_definition == "non-medical":
            response = llm_response(prompt=input.query)
        else:
            points = hybrid_search(query=input.query, collection_name=task_definition, limit=5)
            context = rrf(points=points, n_points=3)
            response = rag_response(context=context, prompt=input.query)

        print("Dynamic response inference took", time.time() - start)
        return {
            "task_definition": task_definition, 
            "response": response
        }
    except Exception as e:
        return {"error": str(e)}