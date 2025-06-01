from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer
import uuid
import os
from datasets import load_dataset
from rich import print
from dotenv import load_dotenv

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

collection_name = "multilingual" # jinaai & multilingual
if not client.collection_exists(collection_name=collection_name):
    print(f"Creating collection {collection_name}...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text-embedding": models.VectorParams(
                size=1024, # 32, 64, 128, 256, 512, 768, 1024
                distance=models.Distance.COSINE
            )
        }
    )
    print(f"Collection {collection_name} created successfully")
else:
    print(f"Collection {collection_name} already exists")

embedder = SentenceTransformer("intfloat/multilingual-e5-large")
# embedder = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

MAX_CHARS = 1500

print("Filtering and truncating texts...")
processed_answers = []
original_answers = []

for answer in answers:
    if len(answer) <= MAX_CHARS:
        processed_answers.append(answer)
        original_answers.append(answer)
    else:
        # Truncate at word boundary
        truncated = answer[:MAX_CHARS]
        last_space = truncated.rfind(' ')
        if last_space > MAX_CHARS * 0.8:  # If we can find a space in the last 20%
            truncated = truncated[:last_space]
        processed_answers.append(truncated)
        original_answers.append(answer)

print(f"Processed {len(processed_answers)} answers")
print(f"Max length: {max(len(a) for a in processed_answers)} chars")
print(f"Average length: {sum(len(a) for a in processed_answers) / len(processed_answers):.1f} chars")

# Process one by one for safety
successful_embeddings = []
successful_texts = []
successful_originals = []

for i, text in enumerate(processed_answers):
    try:
        print(f"Processing {i+1}/{len(processed_answers)}", end='\r')
        embedding = embedder.encode([text], convert_to_numpy=True)
        successful_embeddings.extend(embedding)
        successful_texts.append(text)
        successful_originals.append(original_answers[i])
    except Exception as e:
        print(f"\nFailed at index {i}: {e}")
        print(f"Text length: {len(text)}")
        continue

print(f"\nSuccessfully processed: {len(successful_embeddings)}/{len(processed_answers)}")

if successful_embeddings:
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector={"text-embedding": embedding.tolist()},
            payload={"text": text, "original_text": original}
        )
        for embedding, text, original in zip(successful_embeddings, successful_texts, successful_originals)
    ]

    print(f"Upserting {len(points)} points to Qdrant...")
    client.upsert(collection_name=collection_name, points=points)
    print("Successfully uploaded embeddings to Qdrant!")
else:
    print("No successful embeddings created!")