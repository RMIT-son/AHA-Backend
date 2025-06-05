from fastapi import APIRouter, HTTPException
from database.schemas import ConversationCreate, Message, Conversation
from database.queries import (
    create_conversation, get_all_conversations,
    get_conversation_by_id, add_message
)
from bson import ObjectId
from fastapi.encoders import jsonable_encoder

router = APIRouter(prefix="/api/conversations", tags=["Conversations"])

@router.post("/", response_model=Conversation)
def create(convo: ConversationCreate):
    result = create_conversation(convo.user_id)
    return jsonable_encoder(result)

@router.get("/", response_model=list[Conversation])
def get_all():
    return jsonable_encoder(get_all_conversations())

@router.get("/{conversation_id}", response_model=Conversation)
def get_by_id(conversation_id: str):
    convo = get_conversation_by_id(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return jsonable_encoder(convo)

@router.post("/{conversation_id}/message", response_model=Conversation)
def send(conversation_id: str, message: Message):
    convo = add_message(conversation_id, message.sender, message.content)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return jsonable_encoder(convo)