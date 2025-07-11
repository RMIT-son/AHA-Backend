import os
from uuid import uuid4
from dotenv import load_dotenv
from typing import List
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import PointStruct, ScoredPoint

# Load environment variables from .env file
load_dotenv()

# Initialize Qdrant async client using environment variables
qdrant_client = AsyncQdrantClient(
    url=os.getenv("QDRANT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY")
)

def embed(text: str) -> tuple[list[float], list[int], list[float]]:
    """
    Generate dense and sparse embeddings for a given text.
    Returns:
        dense_vec: List of floats representing dense embedding
        indices: List of ints for sparse embedding indices
        values: List of floats for sparse embedding values
    """
    from app.modules.text_processing.embedders import (
        compute_dense_vector,
        compute_sparse_vector,
    )
    try:
        dense_vec = compute_dense_vector(text)
        indices, values = compute_sparse_vector(text)
        return dense_vec, indices, values
    except Exception as e:
        print(f"[Embedding Error] Failed to embed text: {text}. Error: {e}")
        return [], [], []

async def get_all_messages(collection_name: str) -> List[ScoredPoint]:
    """
    Retrieve all messages for a user from the Qdrant collection.

    Args:
        collection_name: Name of the Qdrant collection

    Returns:
        List of ScoredPoint objects for the user
    """
    try:
        scrolled_points, _ = await qdrant_client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True
        )
        return scrolled_points
    except Exception as e:
        print(f"[Qdrant] Error fetching messages for user {collection_name}: {e}")
        return []

async def remove_oldest_message(existing_messages: list, collection_name: str):
    """
    Remove the oldest message based on timestamp from a specific Qdrant collection.
    Args:
        existing_messages: List of message points
        collection_name: Name of the Qdrant collection
    """
    if not existing_messages:
        return
    try:
        # Sort messages by timestamp and select the oldest
        oldest = sorted(existing_messages, key=lambda p: p.payload["timestamp"])[0]
        await qdrant_client.delete(
            collection_name=collection_name,
            points_selector={"points": [oldest.id]}
        )
    except Exception as e:
        print(f"[Qdrant] Error removing oldest message from {collection_name}: {e}")

async def ensure_collection_exists(collection_name: str):
    """
    Ensure the Qdrant collection exists, create it if not.
    Args:
        collection_name: Name of the Qdrant collection based on user ID
    """
    try:
        if not await qdrant_client.collection_exists(collection_name=collection_name):
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
    except Exception as e:
        print(f"Error: {e}")

async def add_message_vector(collection_name: str, conversation_id: str, user_message: str, bot_response: str, timestamp: str) -> None:
    """
    Embed the user message and insert it into Qdrant.
    If more than 50 messages exist for the user, remove the oldest one first.
    Args:
        collection_name: Name of the Qdrant collection
        conversation_id: The conversation's unique identifier
        user_message: The user's message text
        bot_response: The assistant's reply
        timestamp: Timestamp of the message
    """
    try:
        # Ensure the collection exists
        await ensure_collection_exists(collection_name)

        # Retrieve existing messages for the user
        existing_messages = await get_all_messages(collection_name)

        # Maintain rolling window of 50 messages per user
        if len(existing_messages) >= 50:
            await remove_oldest_message(existing_messages, collection_name)

        # Generate dense and sparse embeddings for the message
        dense_vector, sparse_indices, sparse_values = embed(user_message)

        # Upsert the message and its embeddings into Qdrant
        await qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid4()),
                    vector={
                        "text-embedding": dense_vector,
                        "sparse-embedding": models.SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                        )
                    },
                    payload={
                        "conversation_id": conversation_id,
                        "timestamp": timestamp,
                        "user_message": user_message,
                        "bot_response": bot_response
                    }
                )
            ]
        )
    except Exception as e:
        print(f"[Qdrant] Error adding message to vector DB: {e}")

async def delete_conversation_vectors(collection_name: str, conversation_id: str):
    """Delete all vectors in Qdrant for a given conversation ID."""
    try:
        # Get ALL points (no filter to avoid index requirement)
        scrolled_points, _ = await qdrant_client.scroll(
            collection_name=collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        # Filter in Python to find matching conversation_id
        matching_point_ids = [
            point.id for point in scrolled_points 
            if point.payload and point.payload.get("conversation_id") == conversation_id
        ]
        
        # Delete by IDs if any found
        if matching_point_ids:
            await qdrant_client.delete(
                collection_name=collection_name,
                points_selector=matching_point_ids,
                wait=True
            )
            
        print(f"Deleted {len(matching_point_ids)} points for conversation {conversation_id}")
        
    except Exception as e:
        print(f"[Qdrant] Error deleting conversation vectors: {e}")
        raise