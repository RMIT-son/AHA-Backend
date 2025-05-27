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

# 1. Load all docs with debug output
loaders = []
loaders.append(DirectoryLoader(folder_path, glob="**/*.pdf", loader_cls=PyPDFLoader))

all_docs = []
for i, loader in enumerate(loaders):
    print(f"Loading with loader {i+1}...")
    docs = loader.load()
    print(f"Loader {i+1} found {len(docs)} documents")
    all_docs.extend(docs)

print(f"Total documents loaded: {len(all_docs)}")

if len(all_docs) == 0:
    print("ERROR: No documents were loaded! Check your file paths and formats.")
    exit(1)

# Print first few docs for debugging
for i, doc in enumerate(all_docs[:3]):  # Show first 3 docs
    print(f"Doc {i+1}: Source: {doc.metadata.get('source', 'unknown')}")
    print(f"Doc {i+1}: Content length: {len(doc.page_content)}")
    print(f"Doc {i+1}: First 100 chars: {doc.page_content[:100]}...")

# 2. Split each doc into chunks with debug output
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
)

all_chunks = []
for i, doc in enumerate(all_docs):
    chunks = text_splitter.split_documents([doc])
    print(f"Document {i+1} split into {len(chunks)} chunks")
    all_chunks.extend(chunks)

print(f"Total chunks created: {len(all_chunks)}")

if len(all_chunks) == 0:
    print("ERROR: No chunks were created! Check your documents have content.")
    exit(1)

# 3. Create collection (if not exists)
try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print("Successfully connected to Qdrant")
except Exception as e:
    print(f"ERROR connecting to Qdrant: {e}")
    exit(1)

collection_name = "test"
print(f"Using collection: {collection_name}")

if not client.collection_exists(collection_name=collection_name):
    print(f"Creating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text-embedding": models.VectorParams(
                size=384, # Dimension of text embeddings
                distance=models.Distance.COSINE # Cosine similarity
            )
        }
    )
    print(f"Collection '{collection_name}' created successfully")
else:
    print(f"Collection '{collection_name}' already exists")

# 4. Embed and prepare Qdrant points
print("Loading embedding model...")
try:
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embedding model loaded successfully")
except Exception as e:
    print(f"ERROR loading embedding model: {e}")
    exit(1)

print("Creating embeddings and points...")
points = []
for i, chunk in enumerate(all_chunks):
    if i % 50 == 0:
        print(f"Processing chunk {i+1}/{len(all_chunks)}")
    
    try:
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
    except Exception as e:
        print(f"ERROR creating embedding for chunk {i}: {e}")
        continue

print(f"Created {len(points)} points for ingestion")

if len(points) == 0:
    print("ERROR: No points were created! Check embedding process.")
    exit(1)

# 5. Ingest points
print("Starting ingestion...")
total = len(points)
batch_size = 100
successful_batches = 0

for i in range(0, total, batch_size):
    batch = points[i:i + batch_size]
    batch_num = i//batch_size + 1
    total_batches = ceil(total/batch_size)
    
    try:
        operation_info = client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"✓ Inserted batch {batch_num}/{total_batches} - Status: {operation_info.status}")
        successful_batches += 1
    except Exception as e:
        print(f"✗ ERROR inserting batch {batch_num}: {e}")

print(f"\nIngestion complete!")
print(f"Successfully inserted {successful_batches}/{ceil(total/batch_size)} batches")
print(f"Total points attempted: {total}")

# 6. Verify ingestion
try:
    collection_info = client.get_collection(collection_name)
    print(f"Collection now contains {collection_info.points_count} points")
except Exception as e:
    print(f"ERROR getting collection info: {e}")