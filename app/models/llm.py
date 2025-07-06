import dspy
from typing import Optional, Union
from app.utils import create_signature_with_doc

class LLMResponse(dspy.Signature):
    recent_conversations: Optional[str] = dspy.InputField(optional=True, description="Recent conversations")
    prompt: str = dspy.InputField()
    image: Optional[Union[str, dspy.Image]] = dspy.InputField(optional=True, description="Image from user")
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

    async def forward(self, image: Optional[dspy.Image] = None, prompt: str = None, recent_conversations: str = None) -> str:
        """
        Generate a model response based on the provided prompt, image, and optional conversation history.

        This method asynchronously invokes the response model (`self.response`) with the given inputs,
        which may include multimodal data (text + image) and previous responses for context.

        Args:
            image (Optional[dspy.Image], optional): An image input, if available (e.g., for visual Q&A or diagnosis).
            prompt (str, optional): The current user prompt or message.
            previous_reponses (str, optional): A string representing previous conversation turns to provide context.

        Returns:
            str: The generated response from the model.
        """
        response = await self.response.acall(prompt=prompt, image=image, recent_conversations=recent_conversations)
        return response.response