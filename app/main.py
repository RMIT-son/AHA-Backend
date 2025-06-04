# === Imports ===
import os
import dspy
import time
import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# === Local Module Imports ===
from database.redis_client import RedisClient
from database.qdrant_client import QdrantRAGClient
from modules.text_processing.LLM import LLM
from modules.text_processing.RAG import RAG
from modules.text_processing.task_definition import TaskClassifier

# === FastAPI App Initialization ===
app = FastAPI()

# === CORS Configuration for Local Frontend Access ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Redis Configuration Retrieval ===
redis_client = RedisClient()
rag_config = json.loads(redis_client.get("rag"))
llm_config = json.loads(redis_client.get("llm"))
task_classifier_config = json.loads(redis_client.get("task_classifier"))

# === Qdrant Client Initialization for RAG ===
qdrant_client = QdrantRAGClient(model_name="intfloat/multilingual-e5-small") # ./models/multilingual-e5-large

# === DSPy LLM Setup ===
lm = dspy.LM(
    model=llm_config["model"],
    base_url=os.getenv("OPEN_ROUTER_URL"),
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    cache=False,
    track_usage=True
)
dspy.settings.configure(lm=lm)

# === LLM Warmup for Faster First Inference ===
_ = lm("Say one word.")
QDRANT_COLLECTION=os.getenv('COLLECTION_NAME')

# === DSPy-based Wrappers Initialization ===
llm = LLM(config=llm_config)
rag = RAG(config=rag_config)
task_classifier = TaskClassifier(config=task_classifier_config)

# === Request Schema ===
class QueryInput(BaseModel):
    query: str  # Input prompt from user

# === Endpoint: Pure LLM Response ===
@app.post("/llm")
def llm_response(input: QueryInput):
    """
    Uses the LLM module directly without any context or classification.
    """
    try:
        start = time.time()
        response = llm.forward(prompt=input.query)
        print("LLM inference took", time.time() - start)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

# === Endpoint: Retrieval-Augmented Generation (RAG) ===
@app.post("/rag")
def rag_response(input: QueryInput):
    """
    Retrieves relevant context from Qdrant and uses RAG to respond.
    """
    try:
        start = time.time()
        context = qdrant_client.hybrid_search(
            question=input.query,
            n_points=3,
            collection_name=QDRANT_COLLECTION
        )
        response = rag.forward(context=context, prompt=input.query)
        print("RAG inference took", time.time() - start)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

# === Endpoint: Task Classification and Conditional Processing ===
@app.post("/task-classification")
def task_response(input: QueryInput):
    """
    First classifies the input query, then dynamically chooses between
    LLM-only or RAG depending on task type (e.g., 'medical').
    """
    try:
        start = time.time()
        task_definition = task_classifier.forward(input.query)
        
        if task_definition == "medical":
            # For medical queries, retrieve external context before responding
            context = qdrant_client.hybrid_search(
                question=input.query,
                n_points=3,
                collection_name=QDRANT_COLLECTION
            )
            response = rag.forward(context=context, prompt=input.query)
        else:
            # For non-medical queries, use LLM only
            response = llm.forward(prompt=input.query)

        print("Task classification inference took", time.time() - start)
        return {
            "task_definition": task_definition, 
            "response": response
        }
    except Exception as e:
        return {"error": str(e)}
