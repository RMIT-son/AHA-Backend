
import uuid
from ranx import fuse, Run
from qdrant_client import models
from database.qdrant_client import qdrant_client
from qdrant_client.conversions import common_types as types
from modules.text_processing.embedders import (
        compute_dense_vector, 
        compute_sparse_vector
    )
from dotenv import load_dotenv

load_dotenv()

async def hybrid_search(query: str = None, collection_name: str = None, limit: int = None) -> list[types.QueryResponse]:
        """
        Perform hybrid search using both dense and sparse vectors with Reciprocal Rank Fusion (RRF) from ranx.
        
        Args:
            query: The search query
            collection_name: Name of the Qdrant collection
            limit: Number of final results to return
        
        Returns:
            List of search results ranked by RRF score
        """
        try:
            # Generate query vectors
            embedded_query = compute_dense_vector(query)
            query_indices, query_values = compute_sparse_vector(query)
            
            # Perform separate searches for dense and sparse vectors
            results = await qdrant_client.query_batch_points(
                collection_name=collection_name,
                requests=[
                    models.QueryRequest(
                        query=embedded_query,
                        using="text-embedding",
                        limit=limit,  # Get more results for better fusion
                        with_payload=True
                    ),
                    models.QueryRequest(
                        query=models.SparseVector(
                            indices=query_indices,
                            values=query_values,
                        ),
                        limit=limit,  # Get more results for better fusion
                        with_payload=True,
                        using="sparse-embedding"
                    ),
                ],
            )
            return results
        except Exception as e:
            return {"error": str(e)}

def rrf(points: list[types.QueryResponse] = None, n_points: int = None) -> str:
        """
        Perform Reciprocal Rank Fusion (RRF) on dense and sparse Qdrant search results
        and return a combined context string from the top-ranked documents.
        
        Args:
            results: A tuple/list of two lists - [dense_results, sparse_results]
            n_points: Number of top documents to include in the final context

        Returns:
            A string containing the concatenated content (text) of the top documents
        """
        try:
            # Separate dense and sparse results
            dense_results = points[0].points
            sparse_results = points[1].points
            
            # Collect all unique document IDs from both dense and sparse results
            all_doc_ids = set()
            all_results = {}
            for result in dense_results + sparse_results:
                doc_id = str(result.id)
                all_doc_ids.add(doc_id)
                all_results[doc_id] = result
            
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
            context = "\n".join([doc.payload.get('text', '') for doc in top_docs])
            
            return context
        except Exception as e:
            return {"error": str(e)}