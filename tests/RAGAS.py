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
from datasets import load_dataset
from modules.text_processing.contextual_responder import ContextualResponder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load the dataset from Hugging Face
dataset = load_dataset("Mreeb/Dermatology-Question-Answer-Dataset-For-Fine-Tuning")
data = dataset['train']

# Prepare question and answer arrays (e.g., first 10 for evaluation)
sample_queries = [item['prompt'] for item in data.select(range(5))]
responses = [item['response'] for item in data.select(range(5))]

# Initialize components
llm = ChatOpenAI(model="gpt-4o")
embeddings = SentenceTransformer("BAAI/bge-large-en-v1.5")
response_generator = ContextualResponder()
evaluation_data = []

# Build the evaluation dataset
for query, reference in zip(sample_queries, responses):
    r = response_generator.rag_response(query)
    response = r["response"]
    context = r["context"]

    # Append to RAGAS evaluation dataset
    evaluation_data.append({
        "user_input": query,
        "retrieved_contexts": [context],
        "response": response,
        "reference": reference
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
