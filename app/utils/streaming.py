import dspy
from app.schemas.message import Message
from app.services.manage_responses import TextHandler, ImageHandler, TextImageHandler

async def generate_response_stream(message: Message):
    try:
        # Determine appropriate handler based on message content
        if message.content and not message.images:
            handler = TextHandler()
            output_stream = await handler.handle_text_response(input_data=message)
        elif message.images and not message.content:
            handler = ImageHandler()
            output_stream = await handler.handle_image_response(input_data=message)
        elif message.content and message.images:
            handler = TextImageHandler()
            output_stream = await handler.handle_text_image_response(input_data=message)
        else:
            yield f"data: ERROR - Empty message content and image\n\n"
            return
        
        # Stream the response output
        async for chunk in output_stream:
            if isinstance(chunk, dspy.streaming.StreamResponse):
                yield f"data: {chunk.chunk}\n\n"
            elif isinstance(chunk, dspy.Prediction):
                yield "data: [DONE]\n\n"
                
    except ValueError as ve:
        yield f"data: ERROR - Invalid input: {str(ve)}\n\n"
    except Exception as e:
        yield f"data: ERROR - Stream processing failed: {str(e)}\n\n"