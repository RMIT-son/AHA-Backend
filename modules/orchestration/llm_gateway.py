import os
import dspy
from dotenv import load_dotenv

load_dotenv()

def set_lm_configure(config: dict):
    dspy.settings.configure(lm=dspy.LM(
            model=config["model"],
            base_url=os.getenv("OPEN_ROUTER_URL"),
            api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            cache=True,
            track_usage=True
        )
    )