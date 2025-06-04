import os
import uuid
import torch
from typing import List
from ranx import fuse, Run
from qdrant_client.http import models
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

class QdrantRAGClient:
    """
    Client for performing retrieval from a Qdrant vector database using sentence embeddings.
    Embedding is performed with a HuggingFace model; results are used for RAG.
    """
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", model_s: str = "naver/splade-cocondenser-ensembledistil"):
        # Load environment variables
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")

        # Initialize Qdrant client and embedding model
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.api_key)
        self.embedder = SentenceTransformer(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_s)
        self.model_s = AutoModelForMaskedLM.from_pretrained(model_s)
        
    def compute_sparse_vector(self, text: str):
        """
        Computes a vector from logits and attention mask using ReLU, log, and max operations.
        """
        tokens = self.tokenizer(text, return_tensors="pt")
        output = self.model_s(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        vec = max_val.squeeze()

        # Convert dense vector to sparse: get non-zero indices and values
        indices = torch.nonzero(vec).squeeze().tolist()
        values = vec[indices].tolist()

        return indices, values
    
    def _rrf(self, results: List[models.PointStruct] = None, n_points: int = None) -> str:
        """
        Perform Reciprocal Rank Fusion (RRF) on dense and sparse Qdrant search results
        and return a combined context string from the top-ranked documents.
        
        Args:
            results: A tuple/list of two lists - [dense_results, sparse_results]
            n_points: Number of top documents to include in the final context

        Returns:
            A string containing the concatenated content (text) of the top documents
        """
        
        # Separate dense and sparse results
        dense_results = results[0]
        sparse_results = results[1]
        
        # Collect all unique document IDs from both dense and sparse results
        all_doc_ids = set()
        for result in dense_results + sparse_results:
            all_doc_ids.add(str(result.id))

        # Map each document ID to its full result object for later retrieval
        all_results = {}
        for result in dense_results + sparse_results:
            all_results[str(result.id)] = result
        
        # Use a synthetic query ID for ranx (only one query at a time)
        query_id = str(uuid.uuid4())

        # Build ranx-style runs for dense and sparse results
        dense_run = {query_id: {}}
        sparse_run = {query_id: {}}

        # Fill score maps for each document ID in both search types
        for doc_id in all_doc_ids:
            # Get dense score if present, else 0.0
            dense_score = next((r.score for r in dense_results if r.id == doc_id), 0.0)
            # Get sparse score if present, else 0.0
            sparse_score = next((r.score for r in sparse_results if r.id == doc_id), 0.0)

            dense_run[query_id][doc_id] = dense_score
            sparse_run[query_id][doc_id] = sparse_score

        # Apply RRF fusion to combine rankings from dense and sparse search
        fused_run = fuse(
            runs=[
                Run(name="dense", run=dense_run),
                Run(name="sparse", run=sparse_run)
            ],
            norm="min-max",  # Normalize scores before fusion
            method="rrf"      # Use Reciprocal Rank Fusion method
        )
        
        # Extract fused scores for this query
        doc_scores = fused_run[query_id]

        # Sort documents by RRF score descending and select top-N
        top_doc_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:n_points]

        # Extract just the document IDs
        top_ids = [doc_id for doc_id, _ in top_doc_ids]

        # Retrieve the full document objects using saved mapping
        top_docs = [all_results[doc_id] for doc_id in top_ids]

        # Concatenate the text fields of the top documents into a single context string
        context = "\n".join([doc.payload['text'] for doc in top_docs])
        
        return context


    def hybrid_search(self, question: str = None, collection_name: str = None, limit: int = 15, n_points: int = 3) -> str:
        """
        Perform hybrid search using both dense and sparse vectors with Reciprocal Rank Fusion (RRF) from ranx.
        
        Args:
            question: The search query
            collection_name: Name of the Qdrant collection
            k: RRF parameter (typically 60, controls how much weight to give to lower-ranked results)
            limit: Number of final results to return
        
        Returns:
            List of search results ranked by RRF score
        """
        # Generate query vectors
        embedded_query = self.embedder.encode(question)
        query_indices, query_values = self.compute_sparse_vector(question)
        
        # Perform separate searches for dense and sparse vectors
        results = self.client.search_batch(
            collection_name=collection_name,
            requests=[
                models.SearchRequest(
                    vector=models.NamedVector(
                        name="text-embedding",
                        vector=embedded_query,
                    ),
                    limit=limit,  # Get more results for better fusion
                    with_payload=True,
                ),
                models.SearchRequest(
                    vector=models.NamedSparseVector(
                        name="sparse-embedding", 
                        vector=models.SparseVector(
                            indices=query_indices,
                            values=query_values,
                        ),
                    ),
                    limit=limit,  # Get more results for better fusion
                    with_payload=True,
                ),
            ],
        )
        return self._rrf(results, n_points)