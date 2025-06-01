from modules.text_processing.RAG import RAG
from database.qdrant_client import QdrantRAGClient
from database.redis_client import RedisClient
import json
from rich import print
from sklearn.metrics.pairwise import cosine_similarity
from ragas import evaluate
from datasets import load_dataset
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from sentence_transformers import SentenceTransformer
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAI
from ragas import EvaluationDataset
from ragas.metrics import (
    context_recall,
    faithfulness,
    FactualCorrectness,
    answer_relevancy,
    context_precision,
)
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Load the full dataset
dataset = load_dataset("Mreeb/Dermatology-Question-Answer-Dataset-For-Fine-Tuning")

# Access the train split (adjust if using another)
samples = dataset["train"]

# Select the first 1000 samples
data = samples.select(range(20))

# Extract questions and answers
questions = data['prompt'][:1]
answers = data['response'][:1]

# Initialize your models
redis_client = RedisClient()
rag_config = json.loads(redis_client.get("rag"))
llm = OpenAI(
    model=rag_config['model'], 
    api_key=os.getenv('OPEN_ROUTER_API_KEY'),
    base_url=os.getenv('OPEN_ROUTER_URL')
)
# embeddings = SentenceTransformer("BAAI/bge-large-en-v1.5")
embeddings = SentenceTransformer("dangvantuan/vietnamese-embedding")
evaluation_data = []

qdrant_client = QdrantRAGClient(model_name="./models/multilingual-e5-large")
rag = RAG(config=rag_config)
# Build the evaluation dataset
for question, answer in tqdm(zip(questions, answers), desc="QnA Processing..."):
    context = qdrant_client.retrieve(question=question, vector_name="text-embedding", n_points=5, collection_name="multilingual")
    response = rag.forward(prompt=question, context=context)

    # Append to RAGAS evaluation dataset
    evaluation_data.append({
        "user_input": question,
        "retrieved_contexts": [context],
        "response": response,
        "reference": answer
    })

# Create EvaluationDataset object
evaluation_dataset = EvaluationDataset.from_list(evaluation_data)

# Wrap LLM and Embeddings for RAGAS
evaluator_llm = LangchainLLMWrapper(llm)

# Create embeddings wrapper for RAGAS
hf_embeddings = SentenceTransformer("intfloat/multilingual-e5-large")
evaluator_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

# Configure metrics with custom LLM and embeddings
metrics = [
    context_recall,
    faithfulness,
    FactualCorrectness(),
    context_precision,
    answer_relevancy
]

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
    embeddings=evaluator_embeddings  # This prevents OpenAI API calls
)

# Print generation quality metrics
print("\n=== Generation Quality Metrics ===")
print(result)

# Calculate cosine similarity for retrieval quality
print("\n=== Retrieval Cosine Similarity ===")
for item in evaluation_data:
    q_emb = embeddings.encode(item["user_input"])
    c_emb = embeddings.encode(item["retrieved_contexts"][0])
    similarity = cosine_similarity([q_emb], [c_emb])[0][0]
    print(f"Query: {item['user_input'][:40]}... | Cosine Similarity: {similarity:.4f}")
