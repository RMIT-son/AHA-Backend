from fastapi import FastAPI
from pydantic import BaseModel
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
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@app.post("/task-classification")
def task_response(input: QueryInput):
    try:
        task_definition = responder.task_response(input.query)
        response = responder.rag_response(input.query) if task_definition == "medical" else responder.llm_response(input.query)
        return {
            "task_definition": task_definition, 
            "response": response
        }
    except Exception as e:
        return {"error": str(e)}