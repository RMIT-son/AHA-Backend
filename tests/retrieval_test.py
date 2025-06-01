from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    context_recall,
    context_precision,
)
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from database.qdrant_client import QdrantRAGClient
from database.redis_client import RedisClient
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAI
from database.qdrant_client import QdrantRAGClient
import os
load_dotenv()

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")

print(f"Qdrant URL: {QDRANT_URL}")

dataset = load_dataset("Mreeb/Dermatology-Question-Answer-Dataset-For-Fine-Tuning")
data = dataset["train"]
training_samples = data.select(range(80))
testing_samples = data.select(range(80, 101))

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
answers = training_samples['response'] 

collection_name = "multilingual"
embedder = SentenceTransformer("intfloat/multilingual-e5-large", trust_remote_code=True) # jinaai/jina-embeddings-v3 & intfloat/multilingual-e5-large
qdrant_client = QdrantRAGClient(model_name="intfloat/multilingual-e5-large")
redis_client = RedisClient()
rag_config = json.loads(redis_client.get("rag"))
llm = OpenAI(
    model=rag_config['model'], 
    api_key=os.getenv('OPEN_ROUTER_API_KEY'),
    base_url=os.getenv('OPEN_ROUTER_URL')
)

evaluation_data = []
latencies = []
eval_data = testing_samples.select(range(20))

# Convert to pandas DataFrame
df = pd.read_csv("evaluation_20_qa.csv")

questions = df["question"].tolist()
responses = df["answer"].tolist()

for question, response in zip(questions, responses):
    start_time = time.perf_counter()
    retrieved_context = qdrant_client.retrieve(question, vector_name="text-embedding", collection_name=collection_name, n_points=20)
    end_time = time.perf_counter()
    
    latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    latencies.append(latency_ms)

    # Append to RAGAS evaluation dataset
    evaluation_data.append({
        "user_input": question,
        "retrieved_contexts": [retrieved_context],
        "reference": response
    })

evaluation_dataset = EvaluationDataset.from_list(evaluation_data)
evaluator_llm = LangchainLLMWrapper(llm)
metrics = [
    context_recall,
    context_precision,
]

evaluator_embeddings = LangchainEmbeddingsWrapper(embedder)
# Set embeddings for metrics that need them
for metric in metrics:
    if hasattr(metric, 'embeddings'):
        metric.embeddings = evaluator_embeddings
    if hasattr(metric, 'llm'):
        metric.llm = evaluator_llm

# Run evaluation with both LLM and embeddings specified
result = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics,
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

# Print generation quality metrics
print("\n=== Generation Quality Metrics ===")
print(result)

# Calculate cosine similarity for retrieval quality
qc_sims = []
rc_sims = []

for item in evaluation_data:
    q_emb = embedder.encode(item["user_input"])
    r_emb = embedder.encode(item["reference"])
    c_emb = embedder.encode(item["retrieved_contexts"][0])

    sim_qc = cosine_similarity([q_emb], [c_emb])[0][0]
    sim_rc = cosine_similarity([r_emb], [c_emb])[0][0]

    qc_sims.append(sim_qc)
    rc_sims.append(sim_rc)

# Compute averages
avg_qc = np.mean(qc_sims)
avg_rc = np.mean(rc_sims)

print("\n=== Average Retrieval Cosine Similarity ===")
print(f"Average Query-Context Similarity (QC): {avg_qc:.4f}")
print(f"Average Reference-Context Similarity (RC): {avg_rc:.4f}")

print("\n=== Retrieval Latency Metrics ===")
print(f"Average Latency: {np.mean(latencies):.2f} ms")
print(f"Median Latency: {np.median(latencies):.2f} ms")
print(f"Max Latency: {np.max(latencies):.2f} ms")
print(f"Min Latency: {np.min(latencies):.2f} ms")