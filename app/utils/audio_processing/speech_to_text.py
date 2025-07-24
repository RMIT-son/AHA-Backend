from faster_whisper import WhisperModel

def transcribe_audio(filename: str="output.wav", model_size: str="large-v3", device: str="cuda") -> str:
    """
    Transcribe the given audio file using Faster-Whisper.
    """
    print("Loading Whisper model...")
    model = WhisperModel(model_size, device=device, compute_type="float16")

    print("Transcribing...")
    segments, _ = model.transcribe(filename, beam_size=5)
    result = "\n".join([segment.text for segment in segments])
    
    return result