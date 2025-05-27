import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

class QdrantRAGClient:
    """
    Client for performing retrieval from a Qdrant vector database using sentence embeddings.
    Embedding is performed with a HuggingFace model; results are used for RAG.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", n_points=5):
        # Load environment variables
        load_dotenv()
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.collection_name = "test"
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.n_points = n_points

        # Initialize Qdrant client and embedding model
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.api_key)
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)

    def query(self, question: str, vector_name: str):
        """
        Embeds the input question and retrieves the top matching points from Qdrant.

        Args:
            question (str): The natural language query.
            vector_name (str): The name of the vector field used in Qdrant.

        Returns:
            results (ScoredPointList): List of points returned by Qdrant based on similarity.
        """
        embedded_query = self.embedder.embed_query(question)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedded_query,
            using=vector_name,
            limit=self.n_points,
        )
        return results