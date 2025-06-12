
from typing import Coroutine, Any
from googletrans import Translator
from googletrans.models import Translated

async def translate_text(text: str = None, src: str ="auto", dest: str = "en") -> Coroutine[Any, Any, Translated]:
    async with Translator() as translator:
        result = await translator.translate(text=text, src=src, dest=dest)
        return result