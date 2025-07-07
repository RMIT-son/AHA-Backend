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

def determine_message_type(message: Message):
    """Determine the type of message based on content and files"""
    has_text = bool(message.content and message.content.strip())
    has_files = bool(message.files and len(message.files) > 0)
    has_legacy_image = bool(message.image)
    
    # Check for images in files
    has_images = False
    if has_files:
        has_images = any(
            file_data.type.startswith('image/') 
            for file_data in message.files
        )
    
    # Determine message type
    if has_text and (has_images or has_legacy_image):
        return "text_image"
    elif has_images or has_legacy_image:
        return "image"
    elif has_text:
        return "text"
    else:
        return "empty"

def extract_first_image(message: Message):
    """Extract the first image from either files or legacy image field"""
    # Check new files field first
    if message.files:
        for file_data in message.files:
            if file_data.type.startswith('image/'):
                return file_data.data  # Return base64 data
    
    # Fall back to legacy image field
    if message.image:
        return message.image
    
    return None

def extract_base64_content(data_url: str) -> str:
    """Extract base64 content from data URL"""
    if "," in data_url:
        return data_url.split(",", 1)[1]
    return data_url

def get_first_image_base64(message: Message) -> str:
    """Get the first image from message as clean base64 string"""
    # Check new files field first
    if message.files:
        for file_data in message.files:
            if file_data.type.startswith('image/'):
                return extract_base64_content(file_data.data)
    
    # Check legacy image field
    if message.image and isinstance(message.image, str):
        return extract_base64_content(message.image)
    
    return None

# Endpoint to create a new conversation for a given user
@router.post("/create/{user_id}", response_model=Conversation)
async def create_conversation_by_user_id(user_id: str, message: Message):
    try:
        # Generate title based on message content
        title = await ResponseManager.summarize(message)
        
        # Create conversation
        result = create_conversation(user_id=user_id, title=title)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating conversation: {str(e)}")

# Endpoint to retrieve all conversations stored in the database
@router.get("/user/{user_id}", response_model=list[Conversation])
def get_all_conversations_by_user_id(user_id: str):
    try:
        conversations = get_all_conversations(user_id)
        return conversations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")

# Endpoint to retrieve a specific conversation by its ID
@router.get("/chat/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str):
    try:
        convo = get_conversation_by_id(conversation_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return convo
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversation: {str(e)}")

@router.delete("/{conversation_id}/user/{user_id}")
async def delete_conversation(conversation_id: str, user_id: str):
    try:
        return await delete_conversation_by_id(conversation_id, user_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")

# Endpoint to update conversation title (rename)
@router.put("/{conversation_id}/rename", response_model=Conversation)
async def rename_conversation(conversation_id: str, request: UpdateConversationRequest):
    """Rename a conversation by updating its title"""
    try:
        updated_convo = update_conversation_title(conversation_id, request.title)
        if not updated_convo:
            raise HTTPException(status_code=404, detail="Conversation not found or could not be updated")
        return updated_convo
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating conversation: {str(e)}")

@router.post("/{conversation_id}/{user_id}/stream")
async def stream_message(conversation_id: str, user_id: str, message: Message):
    """Stream AI response for a message with optional file attachments"""
    
    async def generate_response_stream():
        try:
            # Log the incoming message for debugging
            print(f"Received message: content='{message.content}', files={len(message.files) if message.files else 0}")
            
            # Determine message type
            message_type = determine_message_type(message)
            print(f"Message type determined: {message_type}")
            
            # Handle different message types
            if message_type == "text":
                print("Processing text message...")
                handler = TextHandler()
                output_stream = await handler.handle_text_response(input_data=message, user_id=user_id)
                
            elif message_type == "image":
                print("Processing image message...")
                handler = ImageHandler()
                # For backward compatibility, set image field if not already set
                if not message.image:
                    message.image = extract_first_image(message)
                    print(f"Set legacy image field: {bool(message.image)}")
                output_stream = await handler.handle_image_response(input_data=message)
                
            elif message_type == "text_image":
                print("Processing text + image message...")
                handler = TextImageHandler()
                # For backward compatibility, set image field if not already set
                if not message.image:
                    message.image = extract_first_image(message)
                    print(f"Set legacy image field: {bool(message.image)}")
                output_stream = await handler.handle_text_image_response(input_data=message, user_id=user_id)
                
            else:
                print("Empty message detected")
                raise ValueError("Empty message content and files")
                
            # Stream the output
            async for chunk in output_stream:
                if isinstance(chunk, dspy.streaming.StreamResponse):
                    yield f"data: {chunk.chunk}\n\n"
                elif isinstance(chunk, dspy.Prediction):
                    yield "data: [DONE]\n\n"
                    # Save the conversation asynchronously
                    asyncio.create_task(
                        add_message(convo_id=conversation_id, message=message, response=chunk.response)
                    )
                    break
                    
        except Exception as e:
            print(f"Error in generate_response_stream: {e}")
            yield f"data: ERROR - {str(e)}\n\n"

    try:
        return StreamingResponse(
            generate_response_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    except Exception as e:
        print(f"Error creating streaming response: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)