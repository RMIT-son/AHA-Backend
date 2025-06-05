import dspy

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
        LLMResponse.__doc__ = config['instruction']
        self.response = dspy.Predict(LLMResponse, temperature=self.temperature, max_tokens=self.max_tokens)
    
    def forward(self, prompt) -> str:
        return self.response(prompt=prompt).response