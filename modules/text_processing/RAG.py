import dspy

class RAGResponse(dspy.Signature):
    context: str = dspy.InputField()
    prompt: str = dspy.InputField()
    response: str = dspy.OutputField()

class RAG(dspy.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.model = config["model"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        RAGResponse.__doc__ = config['instruction']
        self.response = dspy.Predict(RAGResponse, temperature=self.temperature, max_tokens=self.max_tokens)
    
    def forward(self, context, prompt):
        return self.response(context=context, prompt=prompt).response