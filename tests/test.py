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
print(f"Collection Name: {COLLECTION_NAME}")

dataset = load_dataset("Mreeb/Dermatology-Question-Answer-Dataset-For-Fine-Tuning")

data = dataset["train"]

samples = data.select(range(1000))

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')

answers = samples['response'] 

if not client.collection_exists(collection_name="derma-answers"):
    print(f"Creating collection derma-answers...")
    client.create_collection(
        collection_name="derma-answers",
        vectors_config={
            "text-embedding": models.VectorParams(
                size=1024, # Dimension of text embeddings
                distance=models.Distance.COSINE # Cosine similarity
            )
        }
    )
    print(f"Collection derma-answers created successfully")
else:
    print(f"Collection derma-answers already exists")

embeddings = embedder.encode(answers, convert_to_numpy=True)
points = [
    models.PointStruct(
        id=str(uuid.uuid4()),
        vector={"text-embedding": embedding},
        payload={"text": answer, "question": samples[i]["prompt"]}
    )
    for i, (embedding, answer) in enumerate(zip(embeddings, answers))
]

client.upsert(collection_name="derma-answers", points=points)