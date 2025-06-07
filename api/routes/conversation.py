from fastapi import APIRouter, HTTPException
from database.schemas import ConversationCreate, Message, Conversation
from database.queries import (
    create_conversation, get_all_conversations,
    get_conversation_by_id, add_message
)
from fastapi.encoders import jsonable_encoder

# Create a router with a common prefix and tag for all conversation-related endpoints
router = APIRouter(prefix="/api/conversations", tags=["Conversations"])

# Endpoint to create a new conversation for a given user
@router.post("/", response_model=Conversation)
def create(convo: ConversationCreate):
    result = create_conversation(convo.user_id)
    return result

# Endpoint to retrieve all conversations stored in the database
@router.get("/", response_model=list[Conversation])
def get_all():
    return get_all_conversations()

# Endpoint to retrieve a specific conversation by its ID
@router.get("/{conversation_id}", response_model=Conversation)
def get_by_id(conversation_id: str):
    convo = get_conversation_by_id(conversation_id)
    if not convo:
        # Raise error if the conversation does not exist
        raise HTTPException(status_code=404, detail="Conversation not found")
    return convo

# Endpoint to send a message in a conversation and get an updated response
@router.post("/{conversation_id}/message", response_model=Conversation)
async def send(conversation_id: str, message: Message): 
    # Adds the user's message and triggers the bot's response if sender is 'user'
    convo = await add_message(conversation_id, message.sender, message.content) 
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    # Convert ObjectId to a JSON-serializable format
    return jsonable_encoder(convo)
