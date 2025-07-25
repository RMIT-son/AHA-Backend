from datetime import datetime
import dspy
import asyncio
from .common import classify_file
from app.schemas.message import Message
from app.api.database import call_add_message_endpoint
from app.services.manage_responses.web_search import web_search, format_search_results
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

async def handle_web_search(conversation_id: str, q: str):
    """ 
    This function is called when a web search is requested within a conversation.
    Args:
        conversation_id (str): The ID of the conversation.
        q (str): The search query.
    Returns:
        dict: A dictionary containing the search results or an error message.
    """
    try:
        # 1. Perform search
        results = await web_search(q, manual_trigger=True)
        # 2. Format assistant response
        assistant_response = format_search_results(results)
        # 3. Build message
        message = Message(
            content=q,
            files=[],
            timestamp=datetime.utcnow()
        )
        # 4. Save conversation message
        await call_add_message_endpoint(conversation_id, message, assistant_response)
        # 5. Return exactly what your conversation expects
        return {"message": message, "assistant_response": assistant_response}
    except Exception as e:
        return {"error": f"Web search failed: {str(e)}"}