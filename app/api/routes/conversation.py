import dspy
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from database.schemas import (
    ConversationCreate, 
    Message, 
    Conversation
)
from database.queries import (
    create_conversation, get_all_conversations,
    get_conversation_by_id, add_message
)
from app.services import ResponseManager

# Create a router with a common prefix and tag for all conversation-related endpoints
router = APIRouter(prefix="/api/conversations", tags=["Conversations"])

# Endpoint to create a new conversation for a given user
@router.post("/", response_model=Conversation)
def create(convo: ConversationCreate):
    result = create_conversation(convo.user_id)
    return result

# Endpoint to retrieve all conversations stored in the database
@router.get("/", response_model=list[Conversation])
def get_all():
    return get_all_conversations()

# Endpoint to retrieve a specific conversation by its ID
@router.get("/{conversation_id}", response_model=Conversation)
def get_by_id(conversation_id: str):
    convo = get_conversation_by_id(conversation_id)
    if not convo:
        # Raise error if the conversation does not exist
        raise HTTPException(status_code=404, detail="Conversation not found")
    return convo

@router.post("/{conversation_id}/stream", response_model=Conversation)
async def stream_message(conversation_id: str, message: Message):
    try:
        async def read_output_stream():
            output_stream = await ResponseManager.handle_dynamic_response(message)
            async for chunk in output_stream:
                if isinstance(chunk, dspy.streaming.StreamResponse):
                    yield f"data: {chunk.chunk}\n\n"
                elif isinstance(chunk, dspy.Prediction):
                    yield f"data: [DONE]\n\n"
                    asyncio.create_task(add_message(convo_id=conversation_id, message=message, response=chunk.response))

        return StreamingResponse(
            read_output_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)