import dspy
from typing import Dict, Any
from app.models import RAG, LLM, Summarizer
from app.api.database.redis_client import get_config
from app.utils.orchestration.llm_gateway import set_lm_configure

class ModelManager:
    """Manages the lifecycle of ML models."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.lm = set_lm_configure(config=get_config("llm"))

    def load_models(self) -> None:
        """
        Load and initialize all required machine learning models.

        This includes:
        - Configuring the DSPy language model environment.
        - Initializing task-specific LLM instances (e.g., responder, RAG, summarizer).
        - Loading dense and sparse embedding models.

        After successful execution, all models are stored in `self.models`.
        """
        print("Loading LLM models...")
        
        # Set LM configuration 
        dspy.settings.configure(lm=self.lm)
        
        # Initialize LLM models
        self.models["llm_responder"] = LLM(config=get_config("llm"))
        self.models["rag_responder"] = RAG(config=get_config("rag"))
        self.models["summarizer"] = Summarizer(config=get_config("summarizer"))
        
        print("All models loaded successfully!")
    
    def get_model(self, model_name: str) -> Any:
        """
        Retrieve a loaded model instance by its name.

        Args:
            model_name (str): The name identifier of the model to retrieve.

        Returns:
            Any: The model instance.

        Raises:
            KeyError: If the model name is not found in `self.models`.
        """
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        return self.models[model_name]
    
    def cleanup_models(self) -> None:
        """
        Release resources and clear all loaded models.

        Useful for graceful shutdowns or reinitialization.
        """
        print("Cleaning up ML models...")
        self.models.clear()
        print("ML models cleaned up!")
    
    def get_history(self):
        """
        Retrieve metadata of the most recent interaction with the LM.

        Returns:
            dict or object: The last entry in the LM's internal history log.
        """
        return self.lm.history[-1]


# Global model manager instance
model_manager = ModelManager()