from modules.text_processing.contextual_responder import ContextualResponder
from dotenv import load_dotenv
import json
from database.redis_client import RedisClient
from modules.orchestration.llm_gateway import LLMClient
load_dotenv()

client = LLMClient()

redis_client = RedisClient()
llm_config = json.loads(redis_client.get("llm"))
responder = ContextualResponder()

print(client.query("what is 1+1?", 
                   llm_config["model"], 
                   llm_config["system_role"],
                   llm_config["max_tokens"],
                   llm_config["temperature"]))

print(responder.llm_response("what is 1+1?"))