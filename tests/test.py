from database.qdrant_client import QdrantRAGClient
from database.redis_client import RedisClient
from modules.orchestration.llm_gateway import GeneralLLM
import json
from rich import print

# Load Configuration
redis_client = RedisClient()
rag_config = json.loads(redis_client.get("rag"))
llm_config = json.loads(redis_client.get("llm"))

# Clients
qdrant_client = QdrantRAGClient(model_name="BAAI/bge-large-en-v1.5")

# Example Usage
question="Can you give me a piece of useful information in your provided context?"
response_generator = GeneralLLM(config=llm_config)
result = response_generator.forward(question)
print(f"[bold blue]Question:[/bold blue] {question}\n")
print(f"[bold red]Reasoning:[/bold red] {result.reasoning}\n[bold green]General LLM Answer:[/bold green] {result.answer}\n")


response_generator_rag = GeneralLLM(config=rag_config)
context = qdrant_client.retrieve(question, vector_name="text-embedding", n_points=3)
prompt = f"""{context}
            
            Question
            {question}
            Answer:"""
result = response_generator_rag.forward(prompt)
print(f"[bold red]Reasoning:[/bold red] {result.reasoning}\n[bold green]RAG Answer:[/bold green] {result.answer}\n")

