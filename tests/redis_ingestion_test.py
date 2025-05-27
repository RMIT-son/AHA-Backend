import redis
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=int(os.getenv("REDIS_PORT")),
            username="default",
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True
)


config = {
    "model": "gpt-4o-mini",
    "temperature": 0.01,
    "max_tokens": 1024,
    "system_role": """You are a query classifier for a medical chatbot system with a knowledge base focused on skin diseases and dermatology.

Route queries to:
- RAG: Medical questions, health concerns, skin conditions, dermatology, symptoms, treatments, medications, or any health-related advice
- LLM: General conversation, non-medical questions, casual chat, weather, jokes, general knowledge

Be generous with routing to RAG - if there's any medical context, choose RAG.

Respond with only "RAG" or "LLM" - nothing else.

Examples:
- "What is acne?" -> RAG
- "I have a rash" -> RAG
- "How to treat dry skin?" -> RAG
- "Should I see a doctor?" -> RAG
- "What's the weather?" -> LLM
- "Tell me a joke" -> LLM
- "How are you?" -> LLM"""
}

client.set("supervisor", json.dumps(config))