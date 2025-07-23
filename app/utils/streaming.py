import dspy
import asyncio
from .common import classify_file
from app.schemas.message import Message
from app.api.database import call_add_message_endpoint
from app.services.manage_responses import TextHandler, ImageHandler, TextImageHandler

async def generate_response_stream(message: Message, user_id: str, conversation_id: str):
    try:
        # Classify files into images and documents
        img_files, doc_files = classify_file(message.files or [])

        # Determine appropriate handler
        if message.content and not img_files:
            handler = TextHandler()
            output_stream = await handler.handle_text_response(input_data=message, user_id=user_id)
        elif img_files and not (message.content or doc_files):
            handler = ImageHandler()
            output_stream = await handler.handle_image_response(input_data=message, user_id=user_id)
        elif img_files and (message.content or doc_files):
            handler = TextImageHandler()
            output_stream = await handler.handle_text_image_response(input_data=message, user_id=user_id)
        else:
            yield f"data: ERROR - Empty message content and file\n\n"
            return
        
        # Stream the response output
        async for chunk in output_stream:
            if isinstance(chunk, dspy.streaming.StreamResponse):
                yield f"data: {chunk.chunk}\n\n"
            elif isinstance(chunk, dspy.Prediction):
                yield "data: [DONE]\n\n"
                # Call add_message endpoint via HTTP
                asyncio.create_task(
                    call_add_message_endpoint(conversation_id=conversation_id, message=message, response=chunk.response)
                )
    except ValueError as ve:
        yield f"data: ERROR - Invalid input: {str(ve)}\n\n"
    except Exception as e:
        yield f"data: ERROR - Stream processing failed: {str(e)}\n\n"