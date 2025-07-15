import dspy
from .llm import LLM
from app.utils import create_signature_with_doc

class Summarize(dspy.Signature):
    input: str = dspy.InputField()
    title: str = dspy.OutputField()

class Summarizer(LLM):
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.summarize_signature_cls = create_signature_with_doc(Summarize, config["instruction"])
        self.summarize = dspy.Predict(self.summarize_signature_cls,temperature=self.temperature,max_tokens=self.max_tokens)

    async def forward(self, input: str = None) -> str:
        title = self.summarize(input=input)
        return title.title
