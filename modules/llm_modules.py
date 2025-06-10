import dspy
import torch
from typing import Literal
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
    
    async def forward(self, prompt: str) -> str:
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
    
    async def forward(self, context: str, prompt: str) -> str:
        response = await self.response.acall(context=context, prompt=prompt)
        return response.response

class Task(dspy.Signature):
    prompt: str = dspy.InputField()
    task: Literal["non-medical", "dermatology"] = dspy.OutputField()

class Classifier(dspy.Module):
    """Model to classify task into medical or non medical"""
    def __init__(self, config: dict):
        super().__init__()
        self.model = config["model"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        Task.__doc__ = config["instruction"]
        self.clasify = dspy.Predict(Task, temperature=self.temperature, max_tokens=self.max_tokens)
    
    async def forward(self, prompt: str) -> str:
        task = await self.clasify.acall(prompt=prompt)
        return task.task