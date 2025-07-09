import dspy
import asyncio
from app.schemas.message import Message
from app.api.database import call_add_message_endpoint
from app.services.manage_responses import TextHandler, ImageHandler, TextImageHandler

async def generate_response_stream(message: Message, user_id: str, conversation_id: str):
    try:
        # Determine appropriate handler based on message content
        if message.content and not message.image:
            handler = TextHandler()
            output_stream = await handler.handle_text_response(input_data=message, user_id=user_id)
        elif message.image and not message.content:
            handler = ImageHandler()
            output_stream = await handler.handle_image_response(input_data=message, user_id=user_id)
        elif message.content and message.image:
            handler = TextImageHandler()
            output_stream = await handler.handle_text_image_response(input_data=message, user_id=user_id)
        else:
            yield f"data: ERROR - Empty message content and image\n\n"
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