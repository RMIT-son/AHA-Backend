from fastapi import APIRouter, HTTPException
from database.schemas import ConversationCreate, Message, Conversation
from database.queries import (
    create_conversation, get_all_conversations,
    get_conversation_by_id, add_message
)
from fastapi.encoders import jsonable_encoder
from app.middleware import logger 

# Create a router with a common prefix and tag for all conversation-related endpoints
router = APIRouter(prefix="/api/conversations", tags=["Conversations"])

# Endpoint to create a new conversation for a given user
@router.post("/", response_model=Conversation)
def create(convo: ConversationCreate):
    logger.info(f"Creating conversation for user_id={convo.user_id}")
    result = create_conversation(convo.user_id)
    return result

# Endpoint to retrieve all conversations stored in the database
@router.get("/", response_model=list[Conversation])
def get_all():
    logger.info("Fetching all conversations")
    return get_all_conversations()

# Endpoint to retrieve a specific conversation by its ID
@router.get("/{conversation_id}", response_model=Conversation)
def get_by_id(conversation_id: str):
    logger.info(f"Fetching conversation with ID={conversation_id}")
    convo = get_conversation_by_id(conversation_id)
    if not convo:
        # Raise error if the conversation does not exist
        logger.warning(f"Conversation {conversation_id} not found")
        raise HTTPException(status_code=404, detail="Conversation not found")
    return convo

# Endpoint to send a message in a conversation and get an updated response
@router.post("/{conversation_id}/message", response_model=Conversation)
async def send(conversation_id: str, message: Message): 
    logger.info(f"Received message from {message.sender} in conversation {conversation_id}")
    # Adds the user's message and triggers the bot's response if sender is 'user'
    convo = await add_message(conversation_id, message.sender, message.content) 
    if not convo:
        logger.warning(f"Conversation {conversation_id} not found when trying to send message")
        raise HTTPException(status_code=404, detail="Conversation not found")
    # Convert ObjectId to a JSON-serializable format
    return jsonable_encoder(convo)
