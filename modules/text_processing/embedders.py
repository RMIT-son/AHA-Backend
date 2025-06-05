from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

dense_vector_embedder = SentenceTransformer("intfloat/multilingual-e5-small")

tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
sparse_vector_embedder = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")

def compute_sparse_vector(text: str):
        """
        Computes a vector from logits and attention mask using ReLU, log, and max operations.
        """
        tokens = tokenizer(text, return_tensors="pt")
        output = sparse_vector_embedder(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        vec = max_val.squeeze()

        # Convert dense vector to sparse: get non-zero indices and values
        indices = torch.nonzero(vec).squeeze().tolist()
        values = vec[indices].tolist()

        return indices, values

def compute_dense_vector(text: str):
        """Embeds text into dense vectors"""
        embedded_text = dense_vector_embedder.encode(text)
        return embedded_text