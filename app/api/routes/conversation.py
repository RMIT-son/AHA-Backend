# from fastapi import APIRouter
# from app.schemas.audio import Audio
# from app.schemas.message import Message
# from app.utils import build_error_response
# from app.utils.streaming import generate_response_stream
# from app.services.manage_responses import ResponseManager
# from fastapi.responses import StreamingResponse, JSONResponse
# from app.utils.audio_processing.speech_to_text import transcribe_audio

# # Create a router with a common prefix and tag for all conversation-related endpoints
# router = APIRouter(prefix="/api/conversations", tags=["Conversations"])

# @router.post("/generate_title")
# async def generate_title(message: Message):
#     """
#     Generate a conversation title based on the user's initial message content or image.

#     This endpoint uses a summarization model to produce a short, descriptive title
#     for the beginning of a new conversation. It supports both text and image inputs.

#     Args:
#         user_id (str): The ID of the user initiating the conversation.
#         request (Request): The incoming HTTP request containing the JSON body.
#             Expected fields in JSON:
#                 - content (str, optional): User's text message.
#                 - files (list, optional): List of files; expects base64-encoded image at files[0].data.
#                 - timestamp (str, optional): Time the message was sent.

#     Returns:
#         JSONResponse: A JSON object with the generated title:
#             {
#                 "title": "Short summary of message or image"
#             }

#     Error Responses:
#         - 400: If user ID is not provided or input is invalid.
#         - 500: If title generation fails due to internal error or model issues.
#     """
#     try:
        
#         title = await ResponseManager.summarize(message)
#         return JSONResponse(content={"title": title}, status_code=200)
#     except Exception as e:
#         return build_error_response(
#             "TITLE_GENERATION_FAILED",
#             f"Failed to generate title: {str(e)}",
#             500
#         )
    
# @router.post("/stream")
# async def stream_message(message: Message):
#     """
#     Stream a response to a user's message (text, image, or both) and update the conversation.

#     Args:
#         conversation_id (str): The ID of the conversation to append the response to.
#         user_id (str): The ID of the user sending the message.
#         message (Message): The message object containing text and/or image.

#     Returns:
#         StreamingResponse: A streamed response via Server-Sent Events (SSE).
#     """
#     try:
        
#         if not message.content and not message.images:
#             return build_error_response(
#                 "INVALID_INPUT",
#                 "Message must contain either text content or image",
#                 400
#             )

#         return StreamingResponse(
#             generate_response_stream(message=message),
#             media_type="text/event-stream",
#             headers={
#                 "Cache-Control": "no-cache",
#                 "Connection": "keep-alive",
#                 "Access-Control-Allow-Origin": "*",
#                 "Access-Control-Allow-Headers": "Cache-Control"
#             }
#         )
        
#     except Exception as e:
#         return build_error_response(
#             "STREAM_INITIALIZATION_FAILED",
#             f"Failed to initialize message stream: {str(e)}",
#             500
#         )

# @router.post("/speech_to_text")
# async def speech_to_text(request: Audio) -> str:
#     """
#     Transcribe the given audio file using Faster-Whisper.

#     Args:
#         request (Audio): Request containing base64-encoded audio data.

#     Returns:
#         str: The transcribed text from the audio file.
#     """
#     return await transcribe_audio(request.audio)