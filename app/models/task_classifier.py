from transformers import pipeline

class Classifier():
    """Zero-shot model to classify task"""
    def __init__(self, config: dict):
        super().__init__()
        self.model = pipeline("zero-shot-classification", model=config["model"])
        self.candidate_labels = config["candidate_labels"]
    
    def forward(self, prompt: str = None) -> str:
        result = self.model(prompt, candidate_labels=self.candidate_labels)
        return result["labels"][0]