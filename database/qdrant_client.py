import os
from uuid import uuid4
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

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

async def get_all_messages(user_id: str, collection_name: str) -> list:
    """
    Retrieve all messages for a user from the Qdrant collection.
    Args:
        user_id: The user's unique identifier
        collection_name: Name of the Qdrant collection
    Returns:
        List of message points for the user
    """
    try:
        result = await qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=100,
            with_payload=True
        )
        return result.points
    except Exception as e:
        print(f"[Qdrant] Error fetching messages for user {user_id}: {e}")
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
        collection_name: Name of the Qdrant collection
    """
    try:
        await qdrant_client.get_collection(collection_name)
    except Exception:
        try:
            # Create collection with dense and sparse vector configs
            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text-embedding": models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse-embedding": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                }
            )
            print(f"[Qdrant] Created collection: {collection_name}")
        except Exception as e:
            print(f"[Qdrant] Failed to create collection: {e}")

async def add_message_vector(user_id: str, conversation_id: str, user_message: str, bot_response: str, timestamp: str, collection_name: str):
    """
    Embed the user message and insert it into Qdrant.
    If more than 50 messages exist for the user, remove the oldest one first.
    Args:
        user_id: The user's unique identifier
        conversation_id: The conversation's unique identifier
        user_message: The user's message text
        bot_response: The assistant's reply
        timestamp: Timestamp of the message
        collection_name: Name of the Qdrant collection
    """
    try:
        # Ensure the collection exists
        await ensure_collection_exists(collection_name)

        # Retrieve existing messages for the user
        existing_messages = await get_all_messages(user_id, collection_name)

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
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "timestamp": timestamp,
                        "bot_response": bot_response
                    }
                )
            ]
        )
    except Exception as e:
        print(f"[Qdrant] Error adding message to vector DB: {e}")