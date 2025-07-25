from pydantic import BaseModel

class Audio(BaseModel):
    """
    Represents an audio file with base64-encoded data.
    
    Attributes:
        audio (str): Base64-encoded audio data.
    """
    audio: str