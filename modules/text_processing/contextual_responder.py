import json
from database.redis_client import RedisClient
from database.qdrant_client import QdrantRAGClient
from modules.orchestration.llm_gateway import GeneralLLM

qdrant_client = QdrantRAGClient(model_name="BAAI/bge-large-en-v1.5")

# Get model configuration from redis
redis_client = RedisClient()
llm_config = json.loads(redis_client.get("llm"))
rag_config = json.loads(redis_client.get("rag"))
task_classifier_config = json.loads(redis_client.get("task_classifier"))

class ContextualResponder():
    """
    Main class responsible for handling context-aware responses using LLM and RAG.
    Configuration for different models is loaded from Redis.
    """
    def __init__(self):
        try:
            self.general_llm = GeneralLLM(config=llm_config)
            self.rag_llm = GeneralLLM(config=rag_config)
            self.task_classifier = GeneralLLM(config=task_classifier_config)
            
        except Exception as e:
            print(f"Error initializing ContextualResponder: {e}")
            raise
    
    def llm_response(self, prompt: str):
        """
        Generate a response using a general-purpose language model based on the LLM config.
        """
        try:
            return self.general_llm.forward(prompt).answer
        except Exception as e:
            print(f"LLM response error: {e}")
            return "Error: No response from model."
    
    def rag_response(self, query):
        """
        Generate a response using Retrieval-Augmented Generation (RAG).
        """
        try:
            context = qdrant_client.retrieve(query, vector_name="text-embedding", n_points=2, collection_name="derma-answers")
            prompt = f"""{context}
            
                        Question
                        {query}
                        Answer:"""
            response = self.rag_llm.forward(prompt)
            return {"response": response.answer, "context": context}
        except Exception as e:
            print(f"RAG response error: {e}")
            return {"response": "Error: No response from model.", "context": ""}

    def task_response(self, prompt):
        """
        Classify or route the task using a task-specific model.
        """
        try:
            return self.task_classifier.forward(prompt).answer
        except Exception as e:
            print(f"Task response error: {e}")
            return "Error: No response from model."