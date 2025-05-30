from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from dotenv import load_dotenv
import os

class Message(BaseModel):
    """
    Represents a single message in a conversation.

    Attributes:
        sender (Literal["user", "assistant"]): The sender of the message.
        content (str): The content of the message.
        timestamp (datetime): The time the message was sent.
    """
    sender: Literal["user", "assistant"]
    content: str
    timestamp: datetime

class Conversation(BaseModel):
    """
    Represents a conversation between a user and the system.

    Attributes:
        id (str): Unique identifier for the conversation.
        user_id (str): Identifier of the user participating in the conversation.
        created_at (datetime): Timestamp when the conversation was created.
        messages (List[Message]): List of messages exchanged in the conversation.
    """
    id: str
    user_id: str
    created_at: datetime
    messages: List[Message] = []

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")
client = AsyncIOMotorClient(MONGODB_URL)

db = client.aha_db
conversations_collection = db.conversations
messages_collection = db.messages
