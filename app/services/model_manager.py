import dspy
from typing import Dict, Any
from database.redis_client import get_config
from app.models import RAG, LLM, Classifier
from app.modules.orchestration.llm_gateway import set_lm_configure
from app.modules import (
    get_dense_embedder, 
    get_sparse_embedder_and_tokenizer
)
from .model_warmup import warmup_all_models


class ModelManager:
    """Manages the lifecycle of ML models."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.lm = set_lm_configure(config=get_config("llm"))

    def load_models(self) -> None:
        """Load and initialize all ML models."""
        print("Loading LLM models...")
        
        # Set LM configuration 
        dspy.settings.configure(lm=self.lm)
        
        # Initialize LLM models
        self.models["llm_responder"] = LLM(config=get_config("llm"))
        self.models["rag_responder"] = RAG(config=get_config("rag"))
        self.models["classifier"] = Classifier(config=get_config("task_classifier"))
        
        # Load embedding models
        self.models["dense_embedder"] = get_dense_embedder()
        self.models["sparse_tokenizer"], self.models["sparse_embedder"] = get_sparse_embedder_and_tokenizer()
        
        print("All models loaded successfully!")
    
    async def warmup_models(self) -> None:
        """Warm up all models with dummy inference."""
        await warmup_all_models(self.models)
    
    def get_model(self, model_name: str) -> Any:
        """Get a specific model by name."""
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        return self.models[model_name]
    
    def cleanup_models(self) -> None:
        """Clean up models and release resources."""
        print("Cleaning up ML models...")
        self.models.clear()
        print("ML models cleaned up!")
    
    def get_history(self):
        """Get history metadata of the last usage"""
        return self.lm.history[-1]


# Global model manager instance
model_manager = ModelManager()