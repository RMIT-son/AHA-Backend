import os
import time
import dspy
import json
from rich import print
from database.redis_client import RedisClient
from database.qdrant_client import QdrantRAGClient
from modules.text_processing.LLM import LLM
from modules.text_processing.RAG import RAG
from modules.text_processing.task_definition import TaskClassifier
from dotenv import load_dotenv
load_dotenv

# Load Configuration
start = time.time()
redis_client = RedisClient()
rag_config = json.loads(redis_client.get("rag"))
llm_config = json.loads(redis_client.get("llm"))
task_classifier_config = json.loads(redis_client.get("task_classifier"))
print("Redis inference took", time.time() - start)

start = time.time()
lm = dspy.LM(
        model="openai/meta-llama/llama-3.3-8b-instruct:free",
        base_url=os.getenv("OPEN_ROUTER_URL"),
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        cache=False,
    )
dspy.settings.configure(lm=lm)
print("DSPY inference took", time.time() - start)

start = time.time()
qdrant_client = QdrantRAGClient(model_name="./models/multilingual-e5-large")
print("Qdrant inference took", time.time() - start)

start = time.time()
_ = lm(
    prompt=".",
    max_tokens=1,
    temperature=0.0,
)
print("hello took", time.time() - start)

prompt = "Hello, who are you?"
start = time.time()
llm = LLM(config=llm_config)
response = llm.forward(prompt)
print(response)
print("LLM inference took", time.time() - start)

prompt = "Hello, the weather is very beatiful today"
start = time.time()
task_classifier = TaskClassifier(config=task_classifier_config)
task = task_classifier.forward(prompt)
print(task)
print("Task inference took", time.time() - start)

prompt = "What are the recommended medications for atopic dermatitis?"
start = time.time()
context = qdrant_client.retrieve(question=prompt, vector_name="text-embedding", n_points=5, collection_name="multilingual")
print("Retrieval inference took", time.time() - start)

start = time.time()
rag = RAG(config=rag_config)
response = rag.forward(context, prompt)
print(response)
print("RAG inference took", time.time() - start)
