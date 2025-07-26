from pydantic import BaseModel
from typing import Optional, List

class Message(BaseModel):   
    content: Optional[str] = None
    images: Optional[List[str]] = None
    context: Optional[str] = None
    recent_conversations: Optional[str] = None