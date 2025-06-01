import dspy
from typing import Literal

class Task(dspy.Signature):
    prompt: str = dspy.InputField()
    task: Literal['medical', 'non-medical'] = dspy.OutputField()

class TaskClassifier(dspy.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.model = config["model"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        Task.__doc__ = config['instruction']
        self.clasify = dspy.Predict(Task, temperature=self.temperature, max_tokens=self.max_tokens)
    
    def forward(self, prompt):
        return self.clasify(prompt=prompt).task