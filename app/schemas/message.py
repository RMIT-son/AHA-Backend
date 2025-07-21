from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Union, List

class FileData(BaseModel):
    name: str
    type: str
    url: str

class Message(BaseModel):
    content: Optional[str] = None
    files: Optional[List[FileData]] = None
    timestamp: Optional[datetime] = None