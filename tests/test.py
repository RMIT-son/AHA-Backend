# Import required libraries
import uuid  # For generating unique IDs for Qdrant points
import os
import time
from rich import print  # For colored console output
from qdrant_client import models  # Qdrant vector DB models
from datasets import load_dataset  # Hugging Face datasets loader
from qdrant_client import QdrantClient  # Qdrant client
from sentence_transformers import SentenceTransformer  # Embedding model
from dotenv import load_dotenv  # To load environment variables from .env

# Load environment variables (Qdrant credentials, collection name)
load_dotenv()

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")

print(f"Qdrant URL: {QDRANT_URL}")

# Load the Dermatology QA dataset from Hugging Face
dataset = load_dataset("Mreeb/Dermatology-Question-Answer-Dataset-For-Fine-Tuning")
data = dataset["train"]

# Split dataset into training and testing
training_samples = data.select(range(800))
testing_samples = data.select(range(800, 1001))

# Connect to Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
answers = training_samples['response']  # Extract answer texts

# Define your collection name and embedding dimension
collection_name = "multilingual-small"  # You can switch this to create other models collection

# Create collection if it doesn't exist
if not client.collection_exists(collection_name=collection_name):
    print(f"Creating collection {collection_name}...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text-embedding": models.VectorParams(
                size=384,  # Embedding size must match your model output
                distance=models.Distance.COSINE
            )
        }
    )
    print(f"Collection {collection_name} created successfully")
else:
    print(f"Collection {collection_name} already exists")

# Load embedding model
embedder = SentenceTransformer("intfloat/multilingual-e5-small")

# Truncate long answers to a max character limit
MAX_CHARS = 600

print("Filtering and truncating texts...")
processed_answers = []
original_answers = []

for answer in answers:
    if len(answer) <= MAX_CHARS:
        processed_answers.append(answer)
        original_answers.append(answer)
    else:
        # Truncate without cutting off mid-word
        truncated = answer[:MAX_CHARS]
        last_space = truncated.rfind(' ')
        if last_space > MAX_CHARS * 0.8:
            truncated = truncated[:last_space]
        processed_answers.append(truncated)
        original_answers.append(answer)

# Show stats about processed data
print(f"Processed {len(processed_answers)} answers")
print(f"Max length: {max(len(a) for a in processed_answers)} chars")
print(f"Average length: {sum(len(a) for a in processed_answers) / len(processed_answers):.1f} chars")

# Embed and store only successfully processed texts
successful_embeddings = []
successful_texts = []
successful_originals = []

start_time = time.perf_counter()

for i, text in enumerate(processed_answers):
    try:
        print(f"Processing {i+1}/{len(processed_answers)}", end='\r')
        # Generate embedding
        embedding = embedder.encode([text], convert_to_numpy=True)
        # Store results
        successful_embeddings.extend(embedding)
        successful_texts.append(text)
        successful_originals.append(original_answers[i])
    except Exception as e:
        # Handle and report failures
        print(f"\nFailed at index {i}: {e}")
        print(f"Text length: {len(text)}")
        continue

end_time = time.perf_counter()
print(f"Embedding time {(end_time - start_time) * 1000} ms")
print(f"\nSuccessfully processed: {len(successful_embeddings)}/{len(processed_answers)}")

# Upsert successfully embedded points to Qdrant
start_time = time.perf_counter()
if successful_embeddings:
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),  # Unique ID for each point
            vector={"text-embedding": embedding.tolist()},  # Named vector
            payload={"text": text, "original_text": original}  # Metadata
        )
        for embedding, text, original in zip(successful_embeddings, successful_texts, successful_originals)
    ]

    print(f"Upserting {len(points)} points to Qdrant...")
    client.upsert(collection_name=collection_name, points=points)
    print("Successfully uploaded embeddings to Qdrant!")
else:
    print("No successful embeddings created!")

end_time = time.perf_counter()
print(f"Upserting time {(end_time - start_time) * 1000} ms")
