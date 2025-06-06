import os
import json
import redis
from dotenv import load_dotenv

load_dotenv()

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    username="default",
    decode_responses=True
)

def get_config(name: str) -> dict:
    config = json.loads(redis_client.get(name=name))
    return config