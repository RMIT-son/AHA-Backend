import redis
import os
from dotenv import load_dotenv

load_dotenv()

class RedisClient:
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
    
    def get(self, configuration):
        return self.client.get(configuration)