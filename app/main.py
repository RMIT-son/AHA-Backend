from fastapi import FastAPI, HTTPException
import uuid
from pydantic import BaseModel
from typing import List, Dict, Optional
from qdrant_client.http.models import PointStruct
from database.qdrant_client import client
from fastapi.middleware.cors import CORSMiddleware
from modules.text_processing.contextual_responder import ContextualResponder

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

responder = ContextualResponder()

class QueryInput(BaseModel):
    query: str

@app.post("/llm")
def llm_response(input: QueryInput):
    try:
        response = responder.llm_response(input.query)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/rag")
def rag_response(input: QueryInput):
    try:
        response = responder.rag_response(input.query)
        return {"response": response["response"], "context": response["context"]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/task-classification")
def task_response(input: QueryInput):
    try:
        task_definition = responder.task_response(input.query)
        response = responder.rag_response(input.query)["response"] if task_definition == "medical" else responder.llm_response(input.query)
        return {
            "task_definition": task_definition, 
            "response": response
        }
    except Exception as e:
        return {"error": str(e)}
    
# Request model for ingesting vectors into a Qdrant collection
class IngestRequest(BaseModel):
    collection_name: str
    vector_size: int
    vectors: List[List[float]]
    payloads: Optional[List[dict]] = None
    batch_size: int

# Endpoint to ingest data into Qdrant
@app.post("/ingest")
def ingest_data(data: IngestRequest):
    try:
        # Prepare Qdrant PointStructs for each vector
        qdrant_points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=data.payloads[i] if data.payloads else None
            ) for i, vec in enumerate(data.vectors)
        ]
        # Upsert points into the specified collection
        client.upsert(collection_name=data.collection_name, points=qdrant_points)
        return {"status": "success", "ingested": len(qdrant_points)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Request model for querying vectors from a Qdrant collection
class QueryRequest(BaseModel):
    collection_name: str
    query_vector: List[float]
    vector_name: str
    n_points: int

# Endpoint to query similar vectors from Qdrant
@app.post("/query")
def query_vectors(data: QueryRequest):
    try:
        # Query Qdrant for similar vectors
        hits = client.query_points(
            collection_name=data.collection_name,
            query_vector=data.query_vector,
            using=data.vector_name,
            limit=data.n_points
        )

        # Format the results
        results = [{
            "id": hit.id,
            "score": hit.score,
            "payload": hit.payload
        } for hit in hits]

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))