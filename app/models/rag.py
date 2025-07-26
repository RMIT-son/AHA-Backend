import dspy
from typing import Optional, List
from .llm import LLM, LLMResponse
from app.utils import create_signature_with_doc

class RAGResponse(LLMResponse):
    context: str = dspy.InputField(description="Context retrieved from the knowledge base")

class RAG(LLM):
    """Model to generate responses based on the retrieved context."""
    predictor_cls = dspy.ChainOfThought

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.signature_cls = create_signature_with_doc(RAGResponse, config["instruction"])
        self.response = self.predictor_cls(self.signature_cls, temperature=self.temperature, max_tokens=self.max_tokens)

    async def forward(self, context: str = None, images: Optional[List[dspy.Image]] = None, prompt: str = None, recent_conversations: str = None) -> str:
        """
        Generate a model response using optional context, prompt, image input, and recent conversations.

        This method asynchronously calls the response model (`self.response.acall`) with the provided
        context (e.g., retrieved knowledge), a text prompt, an optional image, and recent conversation history.

        Args:
            context (str, optional): Retrieved context from knowledge base to provide background information.
            image (str | dspy.Image, optional): An image input (can be a URL, file path, or `dspy.Image` object).
            prompt (str, optional): The current user message or question.
            recent_conversations (str, optional): Recent conversation history for context.

        Returns:
            str: The generated textual response from the model.
        """
        response = await self.response.acall(
            context=context, 
            prompt=prompt, 
            images=images, 
            recent_conversations=recent_conversations
        )
        return response.response