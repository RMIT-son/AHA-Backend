import dspy
from typing import Optional

class RAGResponse(dspy.Signature):
    context: str = dspy.InputField(description="Context retrieved from the knowledge base")
    image: Optional[str | dspy.Image] = dspy.InputField(optional=True, description="What disease is this image look like?")
    prompt: str = dspy.InputField()
    response: str = dspy.OutputField()

class RAG(dspy.Module):
    """Model to generate responses based on the retrieved context"""
    def __init__(self, config: dict = None):
        super().__init__()
        self.model = config["model"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        RAGResponse.__doc__ = config["instruction"]
        self.response = dspy.ChainOfThought(RAGResponse, temperature=self.temperature, max_tokens=self.max_tokens)

    async def forward(self, context: str = None, image: Optional[str] = None, prompt: str = None) -> str:
        response = await self.response.acall(context=context, prompt=prompt, image=image)
        return response.response