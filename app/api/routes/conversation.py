import dspy
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from database.schemas import (
    Message, 
    Conversation
)
from database.queries import (
    create_conversation, get_all_conversations,
    get_conversation_by_id, add_message
)
from app.services.manage_responses import TextHandler, ImageHandler, TextImageHandler

# Create a router with a common prefix and tag for all conversation-related endpoints
router = APIRouter(prefix="/api/conversations", tags=["Conversations"])

# Endpoint to create a new conversation for a given user
@router.post("/create/{user_id}", response_model=Conversation)
def create_conversation_by_user_id(user_id: str):
    result = create_conversation(user_id)
    return result

# Endpoint to retrieve all conversations stored in the database
@router.get("/user/{user_id}", response_model=list[Conversation])
def get_all_conversations_by_user_id(user_id: str):
    conversations = get_all_conversations(user_id)
    return conversations

# Endpoint to retrieve a specific conversation by its ID
@router.get("/chat/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str):
    convo = get_conversation_by_id(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return convo


@router.post("/{conversation_id}/stream", response_model=Conversation)
async def stream_message(conversation_id: str, message: Message):
    async def generate_response_stream():
        try:
            
            # Determine which handler to use based on message content
            if message.content and not message.image:
                handler = TextHandler()
                output_stream = await handler.handle_text_response(input_data=message)
            elif message.image and not message.content:
                handler = ImageHandler()
                output_stream = await handler.handle_image_response(input_data=message)
            elif message.content and message.image:
                handler = TextImageHandler()
                output_stream = await handler.handle_text_image_response(input_data=message)
            else:
                raise ValueError("Empty message content and image")

            # Stream the output
            async for chunk in output_stream:
                if isinstance(chunk, dspy.streaming.StreamResponse):
                    yield f"data: {chunk.chunk}\n\n"
                elif isinstance(chunk, dspy.Prediction):
                    yield "data: [DONE]\n\n"
                    asyncio.create_task(
                        add_message(convo_id=conversation_id, message=message, response=chunk.response)
                    )
        except Exception as e:
            yield f"data: ERROR - {str(e)}\n\n"

    try:
        return StreamingResponse(
            generate_response_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/{conversation_id}/stream/text", response_model=Conversation)
async def stream_message(conversation_id: str, message: Message):
    async def generate_response_stream():
        try:
            
            handler = TextHandler()
            output_stream = await handler.handle_text_response(input_data=message)

            # Stream the output
            async for chunk in output_stream:
                if isinstance(chunk, dspy.streaming.StreamResponse):
                    yield f"data: {chunk.chunk}\n\n"
                elif isinstance(chunk, dspy.Prediction):
                    yield "data: [DONE]\n\n"
                    asyncio.create_task(
                        add_message(convo_id=conversation_id, message=message, response=chunk.response)
                    )
        except Exception as e:
            yield f"data: ERROR - {str(e)}\n\n"

    try:
        return StreamingResponse(
            generate_response_stream(),
            media_type="text/event-stream",
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@router.post("/{conversation_id}/stream/image", response_model=Conversation)
async def stream_message(conversation_id: str, message: Message):
    async def generate_response_stream():
        try:
            
            handler = ImageHandler()
            output_stream = await handler.handle_image_response(input_data=message)

            # Stream the output
            async for chunk in output_stream:
                if isinstance(chunk, dspy.streaming.StreamResponse):
                    yield f"data: {chunk.chunk}\n\n"
                elif isinstance(chunk, dspy.Prediction):
                    yield "data: [DONE]\n\n"
                    asyncio.create_task(
                        add_message(convo_id=conversation_id, message=message, response=chunk.response)
                    )
        except Exception as e:
            yield f"data: ERROR - {str(e)}\n\n"

    try:
        return StreamingResponse(
            generate_response_stream(),
            media_type="text/event-stream",
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)