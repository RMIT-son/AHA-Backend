from pathlib import Path
from app.api.database.redis_client import get_config
from app.utils.orchestration.llm_gateway import client

def generate_audio(text: str):
    speech_file_path = Path(__file__).parent / "speech.mp3"

    with client.audio.speech.with_streaming_response.create(
        model=get_config("tts_config").get("model"),
        voice=get_config("tts_config").get("voice"),
        input=text,
        instructions=get_config("tts_config").get("instructions"),
    ) as response:
        response.stream_to_file(speech_file_path)