import dspy
from typing import Optional

class LLMResponse(dspy.Signature):
    """Signature for general LLM responses."""
    prompt: str = dspy.InputField()
    image: Optional[str | dspy.Image] = dspy.InputField(optional=True, description="Optional image input for multimodal LLMs")
    response: str = dspy.OutputField()

class LLM(dspy.Module):
    """Model to generate general LLM reponses"""
    signature_cls = LLMResponse
    predictor_cls = dspy.Predict

    def __init__(self, config: dict = None):
        self.model = config["model"]
        self.temperature = config["temperature"]    
        self.max_tokens = config["max_tokens"]
        self.response = self.predictor_cls(self.signature_cls, temperature=self.temperature, max_tokens=self.max_tokens)

    async def forward(self, image: Optional[dspy.Image] = None, prompt: str = None) -> str:
        response = await self.response.acall(prompt=prompt, image=image)
        return response.response