import os
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient

load_dotenv()

qdrant_client = AsyncQdrantClient(
    url=os.getenv("QDRANT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY")
)