import os
import uuid
from math import ceil
from rich import print
from dotenv import load_dotenv
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse
from database.qdrant_client import qdrant_client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from app.modules import compute_dense_vector, compute_sparse_vector
from rich.progress import track

load_dotenv()

def load_documents(folder_path):
    if not os.path.exists(folder_path):
        print(f"[red]ERROR: Folder '{folder_path}' does not exist![/red]")
        exit(1)

    print(f"Folder exists: {folder_path}")
    print(f"Files in folder: {os.listdir(folder_path)}")

    loaders = [
        DirectoryLoader(folder_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=lambda path: TextLoader(path, encoding="utf-8")),
    ]

    all_docs = []
    for i, loader in enumerate(loaders):
        print(f"Loading with loader {i + 1}...")
        docs = list(track(loader.load(), description=f"📄 Loader {i + 1} loading docs"))
        print(f"Loader {i + 1} found {len(docs)} documents")
        all_docs.extend(docs)

    if not all_docs:
        print("[red]ERROR: No documents were loaded![/red]")
        exit(1)

    return all_docs


def split_documents(documents, chunk_size=100, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []

    for i, doc in enumerate(track(documents, description="🔪 Splitting documents")):
        chunks = splitter.split_documents([doc])
        print(f"Document {i + 1} split into {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[red]ERROR: No chunks created![/red]")
        exit(1)

    return all_chunks



async def ensure_collection(collection_name):
    if not await qdrant_client.collection_exists(collection_name=collection_name):
        print(f"[yellow]Creating collection: {collection_name}...[/yellow]")
        await qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "text-embedding": models.VectorParams(size=384, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse-embedding": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
        print(f"[green]Collection '{collection_name}' created successfully.[/green]")
    else:
        print(f"[cyan]Collection '{collection_name}' already exists.[/cyan]")


def embed_chunks(chunks):
    points = []
    for i, chunk in enumerate(track(chunks, description="🧠 Embedding chunks")):
        try:
            embedding = compute_dense_vector(chunk.page_content)
            indices, values = compute_sparse_vector(chunk.page_content)
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "text-embedding": embedding,
                    "sparse-embedding": models.SparseVector(indices=indices, values=values),
                },
                payload={
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", "unknown"),
                },
            )
            points.append(point)
        except Exception as e:
            print(f"[red]ERROR embedding chunk: {e}[/red]")

    if not points:
        print("[red]ERROR: No points created![/red]")
        exit(1)

    return points

async def ingest_points(collection_name, points, batch_size=100):
    print("Starting ingestion...")
    total = len(points)
    successful_batches = 0
    total_batches = ceil(total / batch_size)

    for i in track(range(0, total, batch_size), description="🚀 Ingesting batches"):
        batch = points[i:i + batch_size]
        batch_num = i // batch_size + 1

        try:
            operation_info = await qdrant_client.upsert(collection_name=collection_name, points=batch)
            print(f"✓ Batch {batch_num}/{total_batches} - Status: {operation_info.status}")
            successful_batches += 1
        except Exception as e:
            print(f"[red]✗ ERROR inserting batch {batch_num}: {e}[/red]")

    print(f"[green]Ingestion complete: {successful_batches}/{total_batches} batches[/green]")



async def verify_collection(collection_name):
    try:
        collection_info = await qdrant_client.get_collection(collection_name)
        print(f"[green]Collection now contains {collection_info.points_count} points[/green]")
    except UnexpectedResponse as e:
        print(f"[red]ERROR getting collection info: {e}[/red]")