import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()
class QdrantRAGClient:
    """
    Client for performing retrieval from a Qdrant vector database using sentence embeddings.
    Embedding is performed with a HuggingFace model; results are used for RAG.
    """
    def __init__(self, model_name: str):
        # Load environment variables
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.collection_name = os.getenv("QDRANT_COLLECTION")
        self.api_key = os.getenv("QDRANT_API_KEY")

        # Initialize Qdrant client and embedding model
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.api_key)
        self.embedder = SentenceTransformer(model_name)

    def retrieve(self, question: str, vector_name: str, n_points: int):
        """
        Embeds the input question and retrieves the top matching points from Qdrant.

        Args:
            question (str): The natural language query.
            vector_name (str): The name of the vector field used in Qdrant.
            n_points (int): Number of most similar points returned
        Returns:
            context (str): Context based on similarity.
        """
        embedded_query = self.embedder.encode(question)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedded_query,
            using=vector_name,
            limit=n_points,
        )
        context = "\n".join(r.payload["text"] for r in results.points)
        return context