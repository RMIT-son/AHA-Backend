import time
from fastapi import APIRouter
from database.schemas import QueryInput
from database.redis_client import get_config
from modules.llm_modules import RAG, LLM, Classifier
from modules.orchestration.llm_gateway import set_lm_configure
from modules.text_processing.rag_engine import (
    hybrid_search,
    rrf
)

router = APIRouter(prefix="/api/response", tags=["Text"])

set_lm_configure(config=get_config("llm"))
llm_responder = LLM(config=get_config("llm"))
rag_responder = RAG(config=get_config("rag"))
classifier = Classifier(config=get_config("task_classifier"))

@router.post("/llm")
def llm(input: QueryInput):
    """
    Uses the LLM module directly without any context or classification.
    """
    try:
        start = time.time()
        response = llm_responder.forward(input.query)
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
        points = hybrid_search(query=input.query, collection_name="dermatology", limit=10)
        context = rrf(points=points, n_points=3)
        response = rag_responder.forward(context=context, prompt=input.query)
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
        task_definition = classifier.forward(input.query)
        
        if task_definition == "non-medical":
            response = llm_responder.forward(prompt=input.query)
        else:
            points = hybrid_search(query=input.query, collection_name=task_definition, limit=10)
            context = rrf(points=points, n_points=3)
            response = rag_responder.forward(context=context, prompt=input.query)

        print("Dynamic response inference took", time.time() - start)
        return {
            "task_definition": task_definition, 
            "response": response
        }
    except Exception as e:
        return {"error": str(e)}