# === Import required libraries ===
import os
import json
import time
import numpy as np
from rich import print
from datasets import load_dataset
from langchain_openai import OpenAI  # LangChain LLM wrapper
from database.redis_client import RedisClient  # Custom Redis client
from database.qdrant_client import QdrantRAGClient  # Custom Qdrant RAG retrieval client
from sentence_transformers import SentenceTransformer  # For embeddings
from sklearn.metrics.pairwise import cosine_similarity  # For similarity scoring
from dotenv import load_dotenv  # Load .env file

# === Load environment variables from .env file ===
load_dotenv()
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

print(f"Qdrant URL: {QDRANT_URL}")

# === Load QA dataset for evaluation ===
dataset = load_dataset("Mreeb/Dermatology-Question-Answer-Dataset-For-Fine-Tuning")
data = dataset["train"]

# Select a small test set (21 samples)
testing_samples = data.select(range(80, 101))

# === Initialize models and clients ===
collection_name = "multilingual"  # Qdrant collection name
model_name = "./models/multilingual-e5-large"  # Local path to embedding model

embedder = SentenceTransformer(model_name)  # Load embedder

qdrant_client = QdrantRAGClient(model_name=model_name)  # Qdrant retrieval wrapper
redis_client = RedisClient()  # Redis wrapper

# Load RAG configuration from Redis
rag_config = json.loads(redis_client.get("rag"))

# Initialize LLM from LangChain using OpenRouter (OpenAI-compatible)
llm = OpenAI(
    model=rag_config['model'], 
    api_key=os.getenv('OPEN_ROUTER_API_KEY'),
    base_url=os.getenv('OPEN_ROUTER_URL')
)

# === Prepare for RAG evaluation ===
evaluation_data = []  # Will store evaluation examples
latencies = []  # To record RAG retrieval times

questions = testing_samples["prompt"]
responses = testing_samples["response"]

# === Main Evaluation Loop ===
for question, response in zip(questions, responses):
    # Measure retrieval latency
    start_time = time.perf_counter()
    retrieved_context = qdrant_client.retrieve(
        question,
        vector_name="text-embedding",
        collection_name=collection_name,
        n_points=5  # Top-5 context retrieval
    )
    end_time = time.perf_counter()
    
    latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    latencies.append(latency_ms)

    # Store question, retrieved context, and ground-truth reference
    evaluation_data.append({
        "user_input": question,
        "retrieved_contexts": [retrieved_context],
        "reference": response
    })

# === Compute Retrieval Quality (Cosine Similarity) ===
qc_sims = []  # Query-Context similarities
rc_sims = []  # Reference-Context similarities

for item in evaluation_data:
    # Embed query, reference, and context
    q_emb = embedder.encode(item["user_input"])
    r_emb = embedder.encode(item["reference"])
    c_emb = embedder.encode(item["retrieved_contexts"][0])  # First retrieved context

    # Cosine similarity: query vs. context, reference vs. context
    sim_qc = cosine_similarity([q_emb], [c_emb])[0][0]
    sim_rc = cosine_similarity([r_emb], [c_emb])[0][0]

    qc_sims.append(sim_qc)
    rc_sims.append(sim_rc)

# === Compute Averages ===
avg_qc = np.mean(qc_sims)
avg_rc = np.mean(rc_sims)

# === Report Results ===
print("\n=== Average Retrieval Cosine Similarity ===")
print(f"Average Query-Context Similarity (QC): {avg_qc:.4f}")
print(f"Average Reference-Context Similarity (RC): {avg_rc:.4f}")

print("\n=== Retrieval Latency Metrics ===")
print(f"Average Latency: {np.mean(latencies):.2f} ms")
