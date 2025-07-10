import dspy
from typing import Optional
from .llm import LLM, LLMResponse
from app.services.utils import create_signature_with_doc
class RAGResponse(LLMResponse):
    context: str = dspy.InputField(description="Context retrieved from the knowledge base")

class RAG(LLM):
    """Model to generate responses based on the retrieved context."""
    predictor_cls = dspy.ChainOfThought

    def __init__(self, config: dict = None):
        self.model = config["model"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]

        # Create a unique signature with the custom instruction
        self.signature_cls = create_signature_with_doc(RAGResponse, config["instruction"])

        self.response = self.predictor_cls(self.signature_cls, temperature=self.temperature, max_tokens=self.max_tokens)

    async def forward(self, context: str = None, image: Optional[str | dspy.Image] = None, prompt: str = None) -> str:
        response = await self.response.acall(context=context, prompt=prompt, image=image)
        return response.response