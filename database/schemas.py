from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from datetime import datetime

class Message(BaseModel):
    sender: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None

class Conversation(BaseModel):
    id: str  # This will receive the converted _id
    user_id: str
    created_at: datetime
    messages: List = []

class ConversationCreate(BaseModel):
    user_id: str

    class Config:
        orm_mode = True

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