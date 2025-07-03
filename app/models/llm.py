import dspy
from typing import Optional
from app.services.utils import create_signature_with_doc

class LLMResponse(dspy.Signature):
    prompt: str = dspy.InputField()
    previous_reponses: str = dspy.InputField(description="Past responses from chat bot")
    image: Optional[str | dspy.Image] = dspy.InputField(optional=True, description="Optional image input for multimodal LLMs")
    response: str = dspy.OutputField()

class LLM(dspy.Module):
    """Model to generate general LLM responses."""
    predictor_cls = dspy.Predict

    def __init__(self, config: dict = None):
        self.model = config["model"]
        self.temperature = config["temperature"]    
        self.max_tokens = config["max_tokens"]

        # Create a unique signature class with a custom docstring
        self.signature_cls = create_signature_with_doc(LLMResponse, config["instruction"])

        self.response = self.predictor_cls(self.signature_cls, temperature=self.temperature, max_tokens=self.max_tokens)

    async def forward(self, image: Optional[dspy.Image] = None, prompt: str = None, previous_reponses: str = None) -> str:
        response = await self.response.acall(prompt=prompt, image=image, previous_reponses=previous_reponses)
        return response.response