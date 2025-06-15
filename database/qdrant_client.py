import os
from uuid import uuid4
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from modules.text_processing.embedders import (
    compute_dense_vector,
    compute_sparse_vector,
)

HISTORY_COLLECTION = "chat-history"
load_dotenv()

# Initialize Qdrant async client
qdrant_client = AsyncQdrantClient(
    url=os.getenv("QDRANT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY")
)


def embed(text: str) -> tuple[list[float], list[int], list[float]]:
    """
    Generate dense and sparse embeddings for a given text.
    """
    dense_vec = compute_dense_vector(text)
    indices, values = compute_sparse_vector(text)
    return dense_vec, indices, values


async def get_all_messages(user_id: str) -> list:
    """
    Retrieve all messages for a user from the Qdrant collection.
    """
    try:
        result = await qdrant_client.scroll(
            collection_name=HISTORY_COLLECTION,
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


async def remove_oldest_message(points: list):
    """
    Remove the oldest message based on timestamp from Qdrant.
    """
    if not points:
        return
    try:
        oldest = sorted(points, key=lambda p: p.payload["timestamp"])[0]
        await qdrant_client.delete(
            collection_name=HISTORY_COLLECTION,
            points_selector={"points": [oldest.id]}
        )
    except Exception as e:
        print(f"[Qdrant] Error removing oldest message: {e}")


async def ensure_collection_exists(collection_name: str):
    """
    Ensure the Qdrant collection exists, create it if not.
    """
    try:
        await qdrant_client.get_collection(collection_name)
    except Exception:
        try:
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


async def add_message_vector(user_id: str, conversation_id: str, user_message: str, bot_response: str, timestamp: str):
    """
    Embed the user message and insert it into Qdrant.
    If more than 50 messages exist for the user, remove the oldest one first.
    """
    try:
        await ensure_collection_exists(HISTORY_COLLECTION)

        # Retrieve existing messages
        existing_messages = await get_all_messages(user_id)

        # Maintain rolling window of 50 messages
        if len(existing_messages) >= 50:
            await remove_oldest_message(existing_messages)

        # Generate embeddings
        dense_vector, sparse_indices, sparse_values = embed(user_message)

        # Upsert into Qdrant with assistant reply in payload
        await qdrant_client.upsert(
            collection_name=HISTORY_COLLECTION,
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
