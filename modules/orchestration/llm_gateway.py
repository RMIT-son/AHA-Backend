import dspy
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

# Define a DSPy signature for question answering
class GenerateGenralLLMAnswer(dspy.Signature):
    question = dspy.InputField()  # Input field for the question
    answer = dspy.OutputField()   # Output field for the generated answer

class GeneralLLM():
    """
    A general-purpose LLM wrapper using DSPy.
    It configures a language model based on the provided settings
    and supports CoT-based answer generation.
    """
    def __init__(self, config):
        super().__init__()
        # Initialize DSPy LM with model name and hyperparameters
        self.lm = dspy.LM(
            model=config["model"],
            model_config={
                'temperature': config["temperature"],
                'max_tokens': config["max_tokens"],
            }
        )
        # Configure DSPy with this model (thread-local override)
        dspy.configure(lm=self.lm)

        # Set the docstring of the signature dynamically (used as instruction)
        GenerateGenralLLMAnswer.__doc__ = config["instruction"]

        # Initialize the Chain of Thought generator using the signature
        self.generate_answer = dspy.ChainOfThought(GenerateGenralLLMAnswer)

    def forward(self, prompt: str):
        """
        Generates an answer from the LLM based on a user prompt.

        Returns:
            str: The model's generated answer or error message.
        """
        try:
            # Generate prediction using the Chain of Thought reasoning
            prediction = self.generate_answer(question=prompt)
            return prediction
        except Exception as e:
            # Print and return an error message if inference fails
            print("API error:", e)
            return "Error: No response from model."
