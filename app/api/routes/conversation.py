import asyncio
from app.api.database.database_interaction import call_create_convo_endpoint
from app.schemas.message import Message, FileData
from fastapi import APIRouter, Body, Form, Request, UploadFile, File
from app.utils import build_error_response, handle_file_processing
from typing import List, Optional
from app.schemas.message import Message
from app.schemas.audio import Audio
from fastapi import APIRouter, Request
from app.utils import build_error_response
from app.utils.audio_processing.speech_to_text import transcribe_audio
from fastapi.responses import StreamingResponse, JSONResponse
from app.utils.streaming import generate_response_stream
from app.services.manage_responses import ResponseManager
import httpx

# Create a router with a common prefix and tag for all conversation-related endpoints
router = APIRouter(prefix="/api/conversations", tags=["Conversations"])

@router.post("/generate_title/{user_id}")
async def generate_title(
    user_id: str,
    content: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None)
):
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
        
        message = await handle_file_processing(content, files)
        title = await ResponseManager.summarize(message)
        
        result = await call_create_convo_endpoint(user_id=user_id, title=title)
        return result

    except Exception as e:
        return build_error_response(
            "TITLE_GENERATION_FAILED",
            f"Failed to generate title: {str(e)}",
            500
        )
    
@router.post("/{conversation_id}/{user_id}/stream")
async def stream_message(
    conversation_id: str,
    user_id: str,
    content: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None)
):
    """
    Stream a response to a user's message and update the conversation.
    
    Args:
        conversation_id (str): Conversation ID.
        user_id (str): User ID.
        message (Message): The structured message (can include files or content).

    Returns:
        StreamingResponse: Streamed assistant response.
    """
    try:
        if not conversation_id or not user_id:
            return build_error_response(
                "INVALID_INPUT",
                "Conversation ID and user ID are required",
                400
            )
        message = await handle_file_processing(content, files)

        if not message.content and not message.files:
            return build_error_response(
                "INVALID_INPUT",
                "Message must contain either text or files",
                400
            )
        if not message:
            return build_error_response(
                code="PROCESSING_ERROR",
                message=f"Error processing files: {str(e)}",
                status=500
            )
        
        return StreamingResponse(
            generate_response_stream(
                message=message,
                user_id=user_id,
                conversation_id=conversation_id
            ),
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

@router.post("/speech_to_text")
async def speech_to_text(request: Audio) -> str:
    """
    Transcribe the given audio file using Faster-Whisper.

    Args:
        request (Audio): Request containing base64-encoded audio data.

    Returns:
        str: The transcribed text from the audio file.
    """
    return await transcribe_audio(request.audio)
