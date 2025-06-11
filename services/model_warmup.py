"""
Model warm-up utilities to preload and test all ML components.
"""

import torch
from typing import Dict, Any
from database import DummyScoredPoint, DummyQueryResponse
from modules import (
    rrf,
    hybrid_search
)

def warmup_embedding_models(ml_models: Dict[str, Any], dummy_text: str = "This is a test query to warm up the models.") -> None:
    """Warm up dense and sparse embedding models."""
    print("Warming up embedding models...")
    
    # Warm up dense embedder
    ml_models["dense_embedder"].encode(dummy_text)
    
    # Warm up sparse embedder
    tokens = ml_models["sparse_tokenizer"](dummy_text, return_tensors="pt")
    with torch.no_grad():
        _ = ml_models["sparse_embedder"](**tokens)
    
    print("Embedding models warmed up!")


async def warmup_llm_models(ml_models: Dict[str, Any]) -> None:
    """Warm up LLM models with dummy inference."""
    print("Warming up LLM models...")
    
    _ = await ml_models["llm_responder"].forward(prompt="Hello")
    _ = await ml_models["rag_responder"].forward(context="Test context", prompt="Hello")
    _ = ml_models["classifier"].forward(prompt="Test classification")
    
    print("LLM models warmed up!")

async def warmup_hybrid_search_function() -> None:
    """Warm up the hybrid search function."""
    _ = await hybrid_search(
        query="What are the common treatments for atopic dermatitis?",
        collection_name="dermatology",
        limit=50
    )
    print("Hybrid search function warmed up!")

def warmup_rrf_function() -> None:
    """Warm up the RRF function with dummy data."""
    print("Warming up RRF function...")
    
    # Create dummy dense search results
    dummy_dense_points = [
        DummyScoredPoint(score=0.9, payload={"text": "Dense result 1"}, id="doc1"),
        DummyScoredPoint(score=0.8, payload={"text": "Dense result 2"}, id="doc2"),
        DummyScoredPoint(score=0.7, payload={"text": "Dense result 3"}, id="doc3"),
        DummyScoredPoint(score=0.6, payload={"text": "Dense result 4"}, id="doc4"),
        DummyScoredPoint(score=0.5, payload={"text": "Dense result 5"}, id="doc5"),
        DummyScoredPoint(score=0.4, payload={"text": "Dense result 6"}, id="doc6"),
        DummyScoredPoint(score=0.3, payload={"text": "Dense result 7"}, id="doc7"),
        DummyScoredPoint(score=0.2, payload={"text": "Dense result 8"}, id="doc8"),
        DummyScoredPoint(score=0.1, payload={"text": "Dense result 9"}, id="doc9"),
        DummyScoredPoint(score=0.1, payload={"text": "Dense result 10"}, id="doc10")

    ]
    
    # Create dummy sparse search results
    dummy_sparse_points = [
        DummyScoredPoint(score=0.85, payload={"text": "Sparse result 1"}, id="doc1"),
        DummyScoredPoint(score=0.75, payload={"text": "Sparse result 2"}, id="doc4"),
        DummyScoredPoint(score=0.65, payload={"text": "Sparse result 3"}, id="doc5"),
        DummyScoredPoint(score=0.85, payload={"text": "Sparse result 1"}, id="doc1"),
        DummyScoredPoint(score=0.75, payload={"text": "Sparse result 2"}, id="doc4"),
        DummyScoredPoint(score=0.65, payload={"text": "Sparse result 3"}, id="doc5"),
        DummyScoredPoint(score=0.85, payload={"text": "Sparse result 1"}, id="doc1"),
        DummyScoredPoint(score=0.75, payload={"text": "Sparse result 2"}, id="doc4"),
        DummyScoredPoint(score=0.65, payload={"text": "Sparse result 3"}, id="doc5")
    ]
    
    # Create dummy points structure
    dummy_points = [
        DummyQueryResponse(points=dummy_dense_points),
        DummyQueryResponse(points=dummy_sparse_points)
    ]
    
    # Warm up RRF function
    _ = rrf(points=dummy_points, n_points=3)
    print("RRF function warmed up!")


async def warmup_all_models(ml_models: Dict[str, Any] = None) -> None:
    """Warm up all models and functions."""
    print("Starting model warm-up process...")
    
    try:
        warmup_embedding_models(ml_models)
        await warmup_llm_models(ml_models)
        await warmup_hybrid_search_function()
        warmup_rrf_function()
        print("All models warmed up successfully!")
    except Exception as e:
        print(f"Warning: Model warm-up failed: {e}")
        raise