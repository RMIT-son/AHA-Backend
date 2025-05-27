import json
from database.qdrant_client import QdrantRAGClient
from database.redis_client import RedisClient
from modules.orchestration.llm_gateway import LLMClient

qdrant_client = QdrantRAGClient()
redis_client = RedisClient()
llm_client = LLMClient()

class ContextualResponder():
    """
    Main class responsible for handling context-aware responses using LLM and RAG.
    Configuration for different models is loaded from Redis.
    """
    def __init__(self):
        # Load model configurations for general LLM, RAG, and task classification
        self.llm_config = json.loads(redis_client.get("llm"))
        self.rag_config = json.loads(redis_client.get("rag"))
        self.task_classifier_config = json.loads(redis_client.get("task_classifier"))
    
    def llm_response(self, prompt:str):
        """
        Generate a response using a general-purpose language model based on the LLM config.
        """
        try:
            llm_client.set_model(self.llm_config["model"])
            llm_client.set_instruction(self.llm_config["instruction"])
            llm_client.set_max_tokens(self.llm_config["max_tokens"])
            llm_client.set_temperature(self.llm_config["temperature"])
            return llm_client.query(prompt)
        except Exception as e:
            print("API error:", e)
            return "Error: No response from model."
    
    def rag_response(self, query):
        """
        Generate a response using Retrieval-Augmented Generation (RAG).
        It first retrieves relevant context from Qdrant, then queries the LLM with that context.
        """
        try:
            # Retrieve relevant documents from Qdrant
            results = qdrant_client.query(query, vector_name="text-embedding")
            # Build context from retrieved documents
            context = "\n".join(r.payload["text"] for r in results.points)
            # Construct the prompt using retrieved context and original query
            prompt = f"""{context}
            
                        Question
                        {query}
                        Answer:"""
            # Set model parameters for RAG-based response
            llm_client.set_model(self.rag_config["model"])
            llm_client.set_instruction(self.rag_config["instruction"])
            llm_client.set_max_tokens(self.rag_config["max_tokens"])
            llm_client.set_temperature(self.rag_config["temperature"])
            # Query the LLM with contextualized prompt
            response = llm_client.query(prompt)
            return response, context
        except Exception as e:
            print("API error:", e)

    def task_response(self, prompt):
        """
        Classify or route the task using a task-specific model.
        Useful for identifying the nature of user intent or task category.
        """
        try:
            # Set model parameters from task classifier config
            llm_client.set_model(self.task_classifier_config["model"])
            llm_client.set_instruction(self.task_classifier_config["instruction"])
            llm_client.set_max_tokens(self.task_classifier_config["max_tokens"])
            llm_client.set_temperature(self.task_classifier_config["temperature"])
            # Query the LLM with the task classification prompt
            return llm_client.query(prompt)
        except Exception as e:
            print("API error:", e)
