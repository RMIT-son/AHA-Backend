import os
import dspy
import time
import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from database.redis_client import RedisClient
from database.qdrant_client import QdrantRAGClient
from modules.text_processing.LLM import LLM
from modules.text_processing.RAG import RAG
from modules.text_processing.task_definition import TaskClassifier

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = RedisClient()
rag_config = json.loads(redis_client.get("rag"))
llm_config = json.loads(redis_client.get("llm"))
task_classifier_config = json.loads(redis_client.get("task_classifier"))

qdrant_client = QdrantRAGClient(model_name="./models/multilingual-e5-large")

# DSPy LLM setup
lm = dspy.LM(
    model=llm_config["model"],
    base_url=os.getenv("OPEN_ROUTER_URL"),
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    cache=False,
)
dspy.settings.configure(lm=lm)

# Warmup LLM with minimal token generation
_ = lm("Say one word.")

llm = LLM(config=llm_config)
rag = RAG(config=rag_config)
task_classifier = TaskClassifier(config=task_classifier_config)
class QueryInput(BaseModel):
    query: str

@app.post("/llm")
async def llm_response(input: QueryInput):
    try:
        start = time.time()
        response = llm.forward(prompt=input.query)
        print("LLM inference took", time.time() - start)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

    
@app.post("/rag")
async def rag_response(input: QueryInput):
    try:
        start = time.time()
        context = qdrant_client.retrieve(
            question=input.query,
            vector_name="text-embedding",
            n_points=5,
            collection_name="multilingual"
        )
        response = rag.forward(context=context, prompt=input.query)
        print("RAG inference took", time.time() - start)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}


@app.post("/task-classification")
async def task_response(input: QueryInput):
    try:
        start = time.time()
        task_definition = task_classifier.forward(input.query)
        
        if task_definition == "medical":
            context = qdrant_client.retrieve(
                question=input.query,
                vector_name="text-embedding",
                n_points=5,
                collection_name="multilingual"
            )
            response = rag.forward(context=context, prompt=input.query)
        else:
            response = llm.forward(prompt=input.query)
            
        print("Task classification inference took", time.time() - start)
        return {
            "task_definition": task_definition, 
            "response": response
        }
    except Exception as e:
        return {"error": str(e)}
