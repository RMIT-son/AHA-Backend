from qdrant_client import QdrantClient
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, CSVLoader, JSONLoader
from qdrant_client import models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import uuid
import os
from math import ceil
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")

folder_path = "data"

# 1. Load all docs
loaders = []
loaders.append(DirectoryLoader(folder_path, glob="**/*.pdf", loader_cls=PyPDFLoader))

all_docs = []
for loader in loaders:
    docs = loader.load()
    all_docs.extend(docs)

# 2. Split each doc into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
)

all_chunks = []
for doc in all_docs:
    chunks = text_splitter.split_documents([doc])
    all_chunks.extend(chunks)

# 3. Create collection (if not exists)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "text-embedding": models.VectorParams(
                size=384, # Dimension of text embeddings
                distance=models.Distance.COSINE # Cosine similarity
            )
        }
    )

# 4. Embed and prepare Qdrant points
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
points = []
for chunk in all_chunks:
    embedding = embedder.embed_query(chunk.page_content)
    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector={"text-embedding": embedding},
        payload={
            "text": chunk.page_content,
            "source": chunk.metadata.get("source", "unknown")
        }
    )
    points.append(point)

# 5. Ingest points
total = len(points)
batch_size = 100
for i in range(0, total, batch_size):
    batch = points[i:i + batch_size]
    operation_info = client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )
    print(f"Inserted batch {i//batch_size + 1}/{ceil(total/batch_size)}")

print(operation_info)