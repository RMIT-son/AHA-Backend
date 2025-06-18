import dspy
from .llm import LLM, LLMResponse
from typing import Optional

class RAGResponse(LLMResponse):
    """Signature for general RAG responses."""
    context: str = dspy.InputField(description="Context retrieved from the knowledge base")

class RAG(LLM):
    """Model to generate responses based on the retrieved context"""
    signature_cls = RAGResponse
    predictor_cls = dspy.ChainOfThought
    
    def __init__(self, config: dict = None):
        super().__init__(config=config)
        self.response = self.predictor_cls(self.signature_cls, temperature=self.temperature, max_tokens=self.max_tokens)

    async def forward(self, context: str = None, image: Optional[str | dspy.Image] = None, prompt: str = None) -> str:
        response = await self.response.acall(context=context, prompt=prompt, image=image)
        return response.response