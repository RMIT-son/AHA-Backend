import sys
import os
import dspy
from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
from rich import print

# 1. Configuration & Data Loading
lm = dspy.LM(
    model='gpt-4o-mini',
    model_config={
        'temperature': 0.7,
        'max_tokens': 300,
        'top_p': 1.0,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0
    }
)

dspy.configure(lm=lm)

# 2. Basic Chatbot
class BasicQA(dspy.Signature):  # A. Signature
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    rationale = dspy.OutputField()
    answer = dspy.OutputField()

print("\n### Generate Response ###\n")
generate_answer = dspy.Predict(BasicQA)
question = "In a patient with metastatic melanoma who is responding to pembrolizumab but develops severe bullous pemphigoid, would you discontinue immunotherapy, and how would you manage the autoimmune skin toxicity without compromising cancer treatment?"
pred = generate_answer(question=question)
print(f"Question: {question}\nPredicted Answer: {pred.answer}")

# 3. Chatbot with Chain of Thought
print("\n### Generate Response with Chain of Thought ###\n")
generate_answer_with_chain_of_thought = dspy.ChainOfThought(BasicQA)
pred = generate_answer_with_chain_of_thought(question=question)
print(f"Question: {question}\nThought: {pred.rationale.split('.', 1)[1].strip()}\nPredicted Answer: {pred.answer}")
