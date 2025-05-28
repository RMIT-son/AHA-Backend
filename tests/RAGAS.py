from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset
from ragas.metrics import (
    context_recall,
    faithfulness,
    FactualCorrectness,
    answer_relevancy,
    context_precision,
)
from modules.text_processing.contextual_responder import ContextualResponder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rich import print
import json
import os

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load the dataset from Hugging Face
file_path = "tests/data/medmcqa_derm.jsonl"
data = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# Prepare question and answer arrays (e.g., first 5 for evaluation)
questions = [item['question'] for item in data[:20]]
answers = [item['answer'] for item in data[:20]]

# Initialize components
llm = ChatOpenAI(model="gpt-4o")
embeddings = SentenceTransformer("BAAI/bge-large-en-v1.5")
response_generator = ContextualResponder()
evaluation_data = []

# Build the evaluation dataset
for question, answer in zip(questions, answers):
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

# Wrap LLM
evaluator_llm = LangchainLLMWrapper(llm)

# Evaluate generation quality
result = evaluate( 
    dataset=evaluation_dataset,
    metrics=[
        context_recall,
        faithfulness,
        FactualCorrectness(),
        context_precision,
        answer_relevancy
    ],
    llm=evaluator_llm
)

# Print generation quality metrics
print("\n=== Generation Quality Metrics ===")
print(result)

# Calculate cosine similarity for retrieval quality
print("\n=== Retrieval Cosine Similarity ===")
for item in evaluation_data:
    q_emb = embeddings.encode(item["user_input"])
    c_emb = embeddings.encode(item["retrieved_contexts"])[0]
    similarity = cosine_similarity([q_emb], [c_emb])[0][0]
    print(f"Query: {item['user_input'][:40]}... | Cosine Similarity: {similarity:.4f}")
