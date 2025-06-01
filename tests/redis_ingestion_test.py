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
    "model": "openai/meta-llama/llama-3.3-8b-instruct:free",
    "temperature": 0.01,
    "max_tokens": 4,
    "system_role": """You are a query classifier for a medical chatbot system with a knowledge base focused on skin diseases and dermatology.

Route queries to:
- medical: Medical questions, health concerns, skin conditions, dermatology, symptoms, treatments, medications, or any health-related advice
- non-medical: General conversation, non-medical questions, casual chat, weather, jokes, general knowledge

Be generous with routing to medical - if there's any medical context, choose medical.

Respond with only "medical" or "non-medical" - nothing else.

Examples:
- "What is acne?" -> medical
- "I have a rash" -> medical
- "How to treat dry skin?" -> medical
- "Should I see a doctor?" -> medical
- "What's the weather?" -> non-medical
- "Tell me a joke" -> non-medical
- "How are you?" -> non-medical"""
}

client.set("task_classifier", json.dumps(config))