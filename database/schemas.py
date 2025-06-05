from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Message(BaseModel):
    sender: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None

class ConversationCreate(BaseModel):
    user_id: str

class Conversation(ConversationCreate):
    id: str
    created_at: datetime
    messages: List[Message] = []

    class Config:
        orm_mode = True

class QueryInput(BaseModel):
    query: str