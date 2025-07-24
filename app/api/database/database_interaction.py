
import httpx
from .redis_client import get_config
from app.schemas.message import Message
from app.utils.common import serialize_image
from qdrant_client.conversions import common_types as types

DATA_URL = get_config("api_keys")["DATA_URL"]

async def get_recent_conversations(
    collection_name: str,
    limit: int = 50,
    base_url: str = DATA_URL
) -> str:
    """
    Calls the /recent_conversations endpoint and returns the conversation string.

    Args:
        collection_name (str): Qdrant collection name.
        limit (int): Number of recent conversations to retrieve.
        base_url (str): The base URL of the FastAPI server.

    Returns:
        str: Formatted conversation string, or error message.
    """
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            response = await client.get(
                "/api/model_query/recent_conversations",
                params={"collection_name": collection_name, "limit": limit}
            )

        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code} - {response.text}")

        return response.json().get("recent_conversations", "")

    except Exception as e:
        print(f"[Client Error] Failed to fetch recent conversations: {e}")
        return ""

async def call_add_message_endpoint(conversation_id: str, message: Message, response: str):
    """
    Call the add_message endpoint via HTTP request.
    """
    try:
        base_url = DATA_URL
        
        # Convert and serialize images
        serialized_files = []
        if message.files:
            for file in message.files:
                file_dict = file.dict()  # convert Pydantic model to dict
                if file.type.startswith("image/"):
                    file_dict["url"] = serialize_image(file.url)
                else:
                    file_dict["url"] = file.url
                serialized_files.append(file_dict)
        
        async with httpx.AsyncClient() as client:
            response_data = await client.post(
                f"{base_url}/api/conversations/{conversation_id}/add_message",
                json={
                    "content": message.content,
                    "files": serialized_files,
                    "timestamp": message.timestamp.isoformat(),
                    "response": response
                },
                timeout=30.0
            )
            
            if response_data.status_code != 200:
                print(f"Failed to add message: {response_data.status_code} - {response_data.text}")
                
    except Exception as e:
        print(f"Error calling add_message endpoint: {str(e)}")

async def call_create_convo_endpoint(user_id: str, title: str):
    """
    Call the add_message endpoint via HTTP request.
    """
    try:
        base_url = DATA_URL
    
        async with httpx.AsyncClient() as client:
            title_response = await client.post(
                f"{base_url}/api/conversations/create/{user_id}",
                json={"title": title},
                timeout=30.0
            )

        if title_response.status_code != 200:
            print(f"Failed to create conversation title: {title_response.status_code} - {title_response.text}")
    
        return title_response.json()
    
    except Exception as e:
        print(f"Error calling create_convo endpoint: {str(e)}")
            
async def call_hybrid_search(
    query: str,
    collection_name: str,
    limit: int,
    base_url: str = DATA_URL
) -> list[types.QueryResponse]:
    """
    Calls the hybrid_search endpoint and returns parsed Qdrant QueryResponses.

    Args:
        query (str): Search query.
        collection_name (str): Name of Qdrant collection.
        limit (int): Number of results to return.
        base_url (str): Base URL of your FastAPI server.

    Returns:
        List[types.QueryResponse]: A list of results from both dense and sparse searches.
    """
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            response = await client.get(
                "/api/model_query/hybrid_search",
                params={"query": query, "collection_name": collection_name, "limit": limit}
            )

        if response.status_code != 200:
            raise Exception(f"Hybrid search failed: {response.status_code} - {response.text}")

        # Parse result
        raw = response.json()
        if not isinstance(raw, list) or len(raw) != 2:
            raise Exception(f"Unexpected hybrid search result: {raw}")

        # Deserialize results using qdrant's native model
        dense_result = types.QueryResponse(**raw[0])
        sparse_result = types.QueryResponse(**raw[1])
        return [dense_result, sparse_result]

    except Exception as e:
        print(f"[Hybrid Search Error] {e}")
        raise