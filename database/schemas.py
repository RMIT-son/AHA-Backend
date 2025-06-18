from datetime import datetime
from pydantic import BaseModel, Field
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
    payload: Dict[str, Union[str, float, int, bool, None]]  # Adjust types based on your real payload structure
    id: Union[str, int]  # Qdrant IDs can be string or int

class DummyQueryResponse(BaseModel):
    """Simulates a Qdrant QueryResponse for testing purposes."""
    points: List[DummyScoredPoint]