from datetime import datetime
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Union

class Message(BaseModel):   
    sender: str
    content: Optional[str] = None
    image: Optional[Union[str, bytes]] = None
    timestamp: Optional[datetime] = None

class Conversation(BaseModel):
    id: str  # This will receive the converted _id
    user_id: str
    created_at: datetime
    messages: List[Message] = Field(default_factory=list)

class ConversationCreate(BaseModel):
    user_id: str

    class Config:
        from_attributes = True

class QueryInput(BaseModel):
    query: str = Field(..., max_length=512, description="User input")

class DummyScoredPoint(BaseModel):
    """Simulates a Qdrant ScoredPoint for testing purposes."""
    score: float
    payload: Dict[str, Union[str, float, int, bool, None]]
    id: Union[str, int]

class DummyQueryResponse(BaseModel):
    """Simulates a Qdrant QueryResponse for testing purposes."""
    points: List[DummyScoredPoint]
    
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