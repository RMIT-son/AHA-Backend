import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

class QdrantRAGClient:
    def __init__(self, model_name="all-MiniLM-L6-v2", n_points=5):
        # Load environment variables
        load_dotenv()
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.collection_name = os.getenv("QDRANT_COLLECTION")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.vector_name = "text-embedding"
        self.n_points = n_points

        # Initialize Qdrant client and embedding model
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.api_key)
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)

    def query(self, question: str):
        embedded_query = self.embedder.embed_query(question)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedded_query,
            using=self.vector_name,
            limit=self.n_points,
        )
        return results