from faster_whisper import WhisperModel

import base64
import tempfile
from faster_whisper import WhisperModel

def transcribe_audio(audio: str, model_size: str = "large-v3", device: str = "cuda") -> str:
    """
    Transcribe base64-encoded audio using Faster-Whisper.
    
    Args:
        audio (str): Base64-encoded audio data (WAV format recommended).
        model_size (str): Size of Whisper model.
        device (str): Device to run model on ("cuda" or "cpu").
    
    Returns:
        str: Transcribed text.
    """
    # Decode base64 to raw bytes
    audio_bytes = base64.b64decode(audio)

    # Save to a temporary .wav file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio.flush()

        print("Loading Whisper model...")
        model = WhisperModel(model_size, device=device, compute_type="float16")

        print("Transcribing...")
        segments, _ = model.transcribe(temp_audio.name, beam_size=5)
        result = "\n".join([segment.text for segment in segments])
    
    return result
