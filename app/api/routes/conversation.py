import dspy
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from database.schemas import (
    Message, 
    Conversation,
    UpdateConversationRequest
)
from database.queries import (
    create_conversation, get_all_conversations,
    get_conversation_by_id, add_message, delete_conversation_by_id, 
    update_conversation_title
)
from app.services.manage_responses import TextHandler, ImageHandler, TextImageHandler, ResponseManager

# Create a router with a common prefix and tag for all conversation-related endpoints
router = APIRouter(prefix="/api/conversations", tags=["Conversations"])

# Endpoint to create a new conversation for a given user
@router.post("/create/{user_id}", response_model=Conversation)
async def create_conversation_by_user_id(user_id: str, message: Message):
    """
    Create a new conversation for a given user.

    Args:
        user_id (str): The ID of the user.
        message (Message): The initial message to generate a conversation title.

    Returns:
        Conversation: The newly created conversation with a generated title.
    """
    title = await ResponseManager.summarize(message)
    result = create_conversation(user_id=user_id, title=title)
    return result


# Endpoint to retrieve all conversations stored in the database
@router.get("/user/{user_id}", response_model=list[Conversation])
def get_all_conversations_by_user_id(user_id: str):
    """
    Retrieve all conversations belonging to a specific user.

    Args:
        user_id (str): The ID of the user.

    Returns:
        list[Conversation]: A list of the user's stored conversations.
    """
    conversations = get_all_conversations(user_id)
    return conversations


# Endpoint to retrieve a specific conversation by its ID
@router.get("/chat/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str):
    """
    Retrieve a conversation by its unique conversation ID.

    Args:
        conversation_id (str): The ID of the conversation.

    Raises:
        HTTPException: If the conversation is not found.

    Returns:
        Conversation: The conversation object matching the given ID.
    """
    convo = get_conversation_by_id(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return convo


@router.delete("/{conversation_id}/user/{user_id}")
async def delete_conversation(conversation_id: str, user_id: str):
    """
    Delete a specific conversation by its ID for a given user.

    Args:
        conversation_id (str): The ID of the conversation to delete.
        user_id (str): The ID of the user who owns the conversation.

    Returns:
        JSONResponse: A success message or error details.
    """
    return await delete_conversation_by_id(conversation_id, user_id)


# Endpoint to update conversation title (rename)
@router.put("/{conversation_id}/rename", response_model=Conversation)
async def rename_conversation(conversation_id: str, request: UpdateConversationRequest):
    """
    Rename a conversation by updating its title.

    Args:
        conversation_id (str): The ID of the conversation to rename.
        request (UpdateConversationRequest): The request containing the new title.

    Raises:
        HTTPException: If the conversation is not found or the update fails.

    Returns:
        Conversation: The updated conversation with the new title.
    """
    try:
        updated_convo = update_conversation_title(conversation_id, request.title)
        if not updated_convo:
            raise HTTPException(status_code=404, detail="Conversation not found or could not be updated")
        return updated_convo
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating conversation: {str(e)}")


@router.post("/{conversation_id}/{user_id}/stream", response_model=Conversation)
async def stream_message(conversation_id: str, user_id: str, message: Message):
    """
    Stream a response to a user's message (text, image, or both) and update the conversation.

    Args:
        conversation_id (str): The ID of the conversation to append the response to.
        user_id (str): The ID of the user sending the message.
        message (Message): The message object containing text and/or image.

    Returns:
        StreamingResponse: A streamed response via Server-Sent Events (SSE).
    """
    async def generate_response_stream():
        try:
            # Determine appropriate handler based on message content
            if message.content and not message.image:
                handler = TextHandler()
                output_stream = await handler.handle_text_response(input_data=message, user_id=user_id)
            elif message.image and not message.content:
                handler = ImageHandler()
                output_stream = await handler.handle_image_response(input_data=message)
            elif message.content and message.image:
                handler = TextImageHandler()
                output_stream = await handler.handle_text_image_response(input_data=message, user_id=user_id)
            else:
                raise ValueError("Empty message content and image")

            # Stream the response output
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

