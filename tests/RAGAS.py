from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from ragas import EvaluationDataset
from ragas.metrics import (
    context_recall,
    faithfulness,
    FactualCorrectness,
    answer_relevancy,
    context_precision,
)
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rich import print
import json
import os
from tqdm import tqdm
from modules.text_processing.contextual_responder import ContextualResponder

# Load environment variables
load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

# file_path = "tests/data/medmcqa_derm.jsonl"
# data = []

# with open(file_path, "r", encoding="utf-8") as f:
#     for line in f:
#         data.append(json.loads(line))

# questions = [item['question'] for item in data[:5]]
# answers = [item['answer'] for item in data[:5]]

import csv

file_path = "tests/data/dermatology_qa_cleaned.csv"
data = []

# Read CSV file
with open(file_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# Extract questions and answers
questions = [item['question'] for item in data[:1]]
answers = [item['answer'] for item in data[:1]]

# Initialize your models
llm = ChatAnthropic(model="claude-3-5-haiku-20241022")
embeddings = SentenceTransformer("BAAI/bge-large-en-v1.5")
response_generator = ContextualResponder()
evaluation_data = []

# Build the evaluation dataset
for question, answer in tqdm(zip(questions, answers), desc="QnA Processing..."):
    r = response_generator.rag_response(question)
    response = r["response"]
    context = r["context"]

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
hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'}  # or 'cuda' if you have GPU
)
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
    c_emb = embeddings.encode(item["retrieved_contexts"][0])  # Fixed indexing
    similarity = cosine_similarity([q_emb], [c_emb])[0][0]
    print(f"Query: {item['user_input'][:40]}... | Cosine Similarity: {similarity:.4f}")
