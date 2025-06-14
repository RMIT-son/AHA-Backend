import torch
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer

_model_d = None
_model_s_tokenizer = None
_model_s_embedder = None

def get_dense_embedder():
    """
    Load dense embedder once
    """
    global _model_d
    if _model_d is None:
        print("Loading dense embedder model...")
        _model_d = SentenceTransformer("huggingface_models/multilingual-e5-small")
    return _model_d

def get_sparse_embedder_and_tokenizer():
    """
    Load sparse embedder and tokenizer once
    """
    global _model_s_tokenizer, _model_s_embedder
    if _model_s_tokenizer is None or _model_s_embedder is None:
        print("Loading sparse embedder model and tokenizer...")
        _model_s_tokenizer = AutoTokenizer.from_pretrained("huggingface_models/splade-cocondenser")
        _model_s_embedder = AutoModelForMaskedLM.from_pretrained("huggingface_models/splade-cocondenser")
    return _model_s_tokenizer, _model_s_embedder

def compute_dense_vector(text: str = None) -> List[float] | np.ndarray:
        """
        Embeds text into dense vectors
        """
        embedder = get_dense_embedder()
        embedded_text = embedder.encode(text)
        return embedded_text

def compute_sparse_vector(text: str = None) -> Tuple[List[int], List[float]]:
        """
        Computes a vector from logits and attention mask using ReLU, log, and max operations.
        """
        tokenizer, embedder = get_sparse_embedder_and_tokenizer()
        tokens = tokenizer(text, return_tensors="pt")
        output = embedder(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        vec = max_val.squeeze()

        # Safely get indices of non-zero values
        indices = torch.nonzero(vec, as_tuple=True)[0].tolist()

        if isinstance(indices, int):  # if single int, convert to list
                indices = [indices]

        # Safely get corresponding values
        values = vec[indices].tolist() if indices else []

        return indices, values
