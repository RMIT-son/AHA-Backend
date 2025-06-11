import dspy
from transformers import pipeline

class LLMResponse(dspy.Signature):
    prompt: str = dspy.InputField()
    response: str = dspy.OutputField()

class LLM(dspy.Module):
    """Model to generate general LLM reponses"""
    def __init__(self, config: dict):
        super().__init__()
        self.model = config["model"]
        self.temperature = config["temperature"]    
        self.max_tokens = config["max_tokens"]
        LLMResponse.__doc__ = config["instruction"]
        self.response = dspy.Predict(LLMResponse, temperature=self.temperature, max_tokens=self.max_tokens)
    
    async def forward(self, prompt: str = None) -> str:
        response = await self.response.acall(prompt=prompt)
        return response.response

class RAGResponse(dspy.Signature):
    context: str = dspy.InputField()
    prompt: str = dspy.InputField()
    response: str = dspy.OutputField()

class RAG(dspy.Module):
    """Model to generate responses based on the retrieved context"""
    def __init__(self, config: dict):
        super().__init__()
        self.model = config["model"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        RAGResponse.__doc__ = config["instruction"]
        self.response = dspy.Predict(RAGResponse, temperature=self.temperature, max_tokens=self.max_tokens)
    
    async def forward(self, context: str = None, prompt: str = None) -> str:
        response = await self.response.acall(context=context, prompt=prompt)
        return response.response

class Classifier():
    """Zero-shot model to classify task"""
    def __init__(self, config: dict):
        super().__init__()
        self.model = pipeline("zero-shot-classification", model=config["model"])
        self.candidate_labels = config["candidate_labels"]
    
    def forward(self, prompt: str = None) -> str:
        result = self.model(prompt, candidate_labels=self.candidate_labels)
        return result["labels"][0]