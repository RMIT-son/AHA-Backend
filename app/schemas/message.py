from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Union

class Message(BaseModel):   
    content: Optional[str] = None
    image: Optional[Union[str, bytes]] = None
    timestamp: Optional[datetime] = None