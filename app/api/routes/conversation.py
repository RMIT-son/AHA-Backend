from app.schemas.message import Message
from fastapi import APIRouter, Request 
from app.utils import build_error_response
from fastapi.responses import StreamingResponse, JSONResponse
from app.utils.streaming import generate_response_stream
from app.services.manage_responses import ResponseManager

# Create a router with a common prefix and tag for all conversation-related endpoints
router = APIRouter(prefix="/api/conversations", tags=["Conversations"])

@router.post("/generate_title/{user_id}")
async def generate_title(user_id: str, request: Request):
    """
    Generate a conversation title based on the user's initial message content or image.

    This endpoint uses a summarization model to produce a short, descriptive title
    for the beginning of a new conversation. It supports both text and image inputs.

    Args:
        user_id (str): The ID of the user initiating the conversation.
        request (Request): The incoming HTTP request containing the JSON body.
            Expected fields in JSON:
                - content (str, optional): User's text message.
                - files (list, optional): List of files; expects base64-encoded image at files[0].data.
                - timestamp (str, optional): Time the message was sent.

    Returns:
        JSONResponse: A JSON object with the generated title:
            {
                "title": "Short summary of message or image"
            }

    Error Responses:
        - 400: If user ID is not provided or input is invalid.
        - 500: If title generation fails due to internal error or model issues.
    """
    try:
        if not user_id:
            return build_error_response(
                "INVALID_INPUT",
                "User ID is required",
                400
            )
        
        body = await request.json()
        image_data = None
        content = None
        
        if "content" in body and isinstance(body["content"], str) and body["content"]:
            content = body.get("content")
            
        if "files" in body and isinstance(body["files"], list) and body["files"]:
            image_data = body["files"][0].get("data")
            
        message = Message(
            content=content,
            image=image_data,
            timestamp=body.get("timestamp")
        )
        
        title = await ResponseManager.summarize(message)
        return JSONResponse(content={"title": title}, status_code=200)
    except Exception as e:
        return build_error_response(
            "TITLE_GENERATION_FAILED",
            f"Failed to generate title: {str(e)}",
            500
        )
    
@router.post("/{conversation_id}/{user_id}/stream")
async def stream_message(conversation_id: str, user_id: str, request: Request):
    """
    Stream a response to a user's message (text, image, or both) and update the conversation.

    Args:
        conversation_id (str): The ID of the conversation to append the response to.
        user_id (str): The ID of the user sending the message.
        message (Message): The message object containing text and/or image.

    Returns:
        StreamingResponse: A streamed response via Server-Sent Events (SSE).
    """
    try:
        if not conversation_id or not user_id:
            return build_error_response(
                "INVALID_INPUT",
                "Conversation ID and user ID are required",
                400
            )
        
        body = await request.json()
        image_data = None
        content = None
        
        if "content" in body and isinstance(body["content"], str) and body["content"]:
            content = body.get("content")
            
        if "files" in body and isinstance(body["files"], list) and body["files"]:
            image_data = body["files"][0].get("data")
            
        message = Message(
            content=content,
            image=image_data,
            timestamp=body.get("timestamp")
        )
        
        if not message:
            return build_error_response(
                "INVALID_INPUT",
                "Message is required",
                400
            )
        
        if not message.content and not message.image:
            return build_error_response(
                "INVALID_INPUT",
                "Message must contain either text content or image",
                400
            )

        return StreamingResponse(
            generate_response_stream(message=message, user_id=user_id, conversation_id=conversation_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
        
    except Exception as e:
        return build_error_response(
            "STREAM_INITIALIZATION_FAILED",
            f"Failed to initialize message stream: {str(e)}",
            500
        )

