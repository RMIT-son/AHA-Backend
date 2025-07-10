from datetime import datetime
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Union

class Message(BaseModel):   
    content: Optional[str] = None
    image: Optional[Union[str, bytes]] = None
    timestamp: Optional[datetime] = None

class Conversation(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: datetime
    messages: List[Message] = Field(default_factory=list)

# Request model (for register)
class UserCreate(BaseModel):
    fullName: str
    email: EmailStr
    password: str
    phone: str

# Used when logging in
class UserLogin(BaseModel):
    email: EmailStr
    password: str
    
class UserResponse(BaseModel):
    id: str
    fullName: str
    email: EmailStr
    phone: str

class UpdateConversationRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="New title for the conversation")

class DummyScoredPoint(BaseModel):
    """Simulates a Qdrant ScoredPoint for testing purposes."""
    score: float
    payload: Dict[str, Union[str, float, int, bool, None]]
    id: Union[str, int]

class DummyQueryResponse(BaseModel):
    """Simulates a Qdrant QueryResponse for testing purposes."""
    points: List[DummyScoredPoint]