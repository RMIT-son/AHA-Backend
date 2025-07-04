
from typing import Coroutine, Any
from googletrans import Translator
from googletrans.models import Translated

async def translate_text(text: str = None, src: str ="auto", dest: str = "en") -> Coroutine[Any, Any, Translated]:
    """
    Automatically translate text from a source language to a target language (default: English).

    This function uses an asynchronous translation service (e.g., `deep-translator` or `googletrans`)
    to detect the source language (if not specified) and translate the input text.

    Args:
        text (str, optional): The input text to translate.
        src (str, optional): The source language code (default: "auto" for automatic detection).
        dest (str, optional): The target language code (default: "en" for English).

    Returns:
        Coroutine[Any, Any, Translated]: The translation result object, which contains:
            - `text`: Translated string
            - `src`: Detected source language
            - `dest`: Target language
            - Other metadata depending on the translation library.

    Raises:
        Exception: If the translation process fails or the service is unavailable.
    """
    async with Translator() as translator:
        result = await translator.translate(text=text, src=src, dest=dest)
        return result