import base64
import tempfile
import os
from app.api.database.redis_client import get_config
from openai import OpenAI

client = OpenAI(api_key=get_config("api_keys")["OPENAI_API_KEY"])

async def transcribe_audio(audio: str) -> str:
    """
    Transcribe base64-encoded audio using OpenAI Whisper API.
    
    Args:
        audio (str): Base64-encoded audio data (WAV or MP3).
    
    Returns:
        str: Transcribed text.
    """
    # Decode base64 to raw bytes
    audio_bytes = base64.b64decode(audio)

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio.flush()
        temp_audio_path = temp_audio.name

    try:
        with open(temp_audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            result = response.text
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    return result
