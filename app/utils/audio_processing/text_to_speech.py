from pathlib import Path
from openai import OpenAI
from app.api.database.redis_client import get_config

api_keys = get_config("api_keys")

def generate_audio(text: str, instruction: str):
    client = OpenAI(api_key=api_keys.get("OPENAI_API_KEY"))
    speech_file_path = Path(__file__).parent / "speech.mp3"

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        instructions=instruction,
    ) as response:
        response.stream_to_file(speech_file_path)