import json
from typing import Literal
from modules.llm_modules import Classifier
from database.redis_client import redis_client

classifier_config = json.loads(redis_client.get("task_classifier"))
classifier = Classifier(classifier_config)

def classify_task(prompt: str) -> Literal['dermatology', 'non-medical']:
    """
    First classifies the input query, then dynamically chooses between
    LLM-only or RAG depending on task type (e.g., 'medical').
    """
    try:
        task_definition = classifier.forward(prompt)
        
        return task_definition
    except Exception as e:
        return {"error": str(e)}