import os
import time
import dspy
import json
from rich import print
from database.redis_client import RedisClient
from database.qdrant_client import QdrantRAGClient
from dotenv import load_dotenv
load_dotenv

# # Load Configuration
redis_client = RedisClient()
rag_config = json.loads(redis_client.get("rag"))
llm_config = json.loads(redis_client.get("llm"))
task_classifier_config = json.loads(redis_client.get("task_classifier"))

start = time.time()
lm = dspy.LM(
        model="openai/gpt-4o-mini",
        base_url=os.getenv("OPEN_ROUTER_URL"),
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        cache=False,
    )
dspy.settings.configure(lm=lm)
print("DSPY inference took", time.time() - start)

# from typing import Literal
# class LLMResponse(dspy.Signature):
#     prompt: str = dspy.InputField()
#     response: str = dspy.OutputField()
# class RAGResponse(dspy.Signature):
#     context: str = dspy.InputField()
#     prompt: str = dspy.InputField()
#     response: str = dspy.OutputField()
# class Task(dspy.Signature):
#     prompt: str = dspy.InputField()
#     task: Literal['medical', 'non-medical'] = dspy.OutputField()

# qdrant_client = QdrantRAGClient(model_name="./models/multilingual-e5-large")
# # Define a simple program that makes multiple LM calls
# class MyProgram(dspy.Module):
#     def __init__(self):
#         Task.__doc__ = task_classifier_config['instruction']
#         self.predict1 = dspy.Predict(Task, temperature=task_classifier_config['temperature'], max_tokens=task_classifier_config['max_tokens'])
#         LLMResponse.__doc__ = llm_config['instruction']
#         self.predict2 = dspy.Predict(LLMResponse, temperature=llm_config['temperature'], max_tokens=llm_config['max_tokens'])
#         RAGResponse.__doc__ = rag_config['instruction']
#         self.predict3 = dspy.Predict(RAGResponse, temperature=rag_config['temperature'], max_tokens=rag_config['max_tokens'])

#     def __call__(self, prompt: str) -> str:
#         task = self.predict1(prompt=prompt).task
#         if task == "medical":
#             context = qdrant_client.retrieve(
#                 question=prompt,
#                 vector_name="text-embedding",
#                 n_points=10,
#                 collection_name="multilingual"
#         )
#             response = self.predict3(prompt=prompt, context=context)
#         else:
#             response = self.predict2(prompt=prompt)
#         return response.response

# # Run the program and check usage
# start = time.time()
# program = MyProgram()
# output = program(prompt="My skin is red and it feels very itchy")
# print("MyProgram inference took", time.time() - start)
# print(output)

def retrieve_dermatology_data(question) -> str:
    """Retrieve context from database based on dermatology"""
    qdrant_client = QdrantRAGClient(model_name="./models/multilingual-e5-large")
    context = qdrant_client.retrieve(
            question=question,
            vector_name="text-embedding",
            n_points=3,
            collection_name="multilingual"
        )
    return context

def get_user_name():
    return "Tran Duc Duy"

def get_user_age():
    return 20

class DSPyAirlineCustomerSerice(dspy.Signature):
    """You are dermatology assistance.

    You are given a list of tools to handle user request, and you should decide the right tool to use in order to
    fullfil users' request."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
                "Message that summarizes the process result, and the information users need"
            )
        )

agent = dspy.ReAct(
    DSPyAirlineCustomerSerice,
    tools = [
        retrieve_dermatology_data,
        get_user_name,
        get_user_age
    ]
)

result = agent(user_request="My skin is red and feels very itchy, what should I do?")
print(result)
