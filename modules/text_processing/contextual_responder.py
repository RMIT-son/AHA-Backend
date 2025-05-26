import json
from database.qdrant_client import QdrantRAGClient
from database.redis_client import RedisClient
from modules.orchestration.llm_gateway import LLMClient

qdrant_client = QdrantRAGClient()
redis_client = RedisClient()
llm_client = LLMClient()

class ContextualResponder():
    def __init__(self):
        self.llm_config = json.loads(redis_client.get("llm"))
        self.rag_config = json.loads(redis_client.get("rag"))
        self.task_classifier_config = json.loads(redis_client.get("task_classifier"))
    
    def llm_response(self, prompt:str):
        try:
            model = self.llm_config["model"]
            instruction = self.llm_config["instruction"]
            max_tokens = self.llm_config["max_tokens"]
            temperature = self.llm_config["temperature"]
            return llm_client.query(prompt, model, instruction, max_tokens, temperature)
        except Exception as e:
            print("API error:", e)
            return "Error: No response from model."
    
    def rag_response(self, query):
        try:
            results = qdrant_client.query(query)
            context = "\n".join(r.payload["text"] for r in results.points)
            prompt = f"""{context}
            
                        Question
                        {query}
                        Answer:"""
            model = self.rag_config["model"]
            instruction = self.rag_config["instruction"]
            max_tokens = self.rag_config["max_tokens"]
            temperature = self.rag_config["temperature"]
            response = llm_client.query(prompt, model, instruction, max_tokens, temperature)
            return response
        except Exception as e:
            print("API error:", e)

    def task_response(self, prompt):
        try:
            model = self.task_classifier_config["model"]
            instruction = self.task_classifier_config["instruction"]
            max_tokens = self.task_classifier_config["max_tokens"]
            temperature = self.task_classifier_config["temperature"]
            return llm_client.query(prompt, model, instruction, max_tokens, temperature)
        except Exception as e:
            print("API error:", e)
