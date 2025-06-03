import redis
import os
from dotenv import load_dotenv

load_dotenv()

class RedisClient:
    """
    A simple Redis client wrapper for fetching configuration values stored in Redis.
    Configuration includes host, port, and password which are loaded from environment variables.
    """
    def __init__(self):
        self.host = os.getenv("REDIS_HOST")
        self.port = int(os.getenv("REDIS_PORT"))
        self.password = os.getenv("REDIS_PASSWORD")
        self.username = "default"
        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password,
            username=self.username,
            decode_responses=True
        )
    
    def get(self, configuration: str) -> dict:
        return self.client.get(configuration)