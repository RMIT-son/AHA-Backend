import dspy
from app.api.database.redis_client import get_config

def set_lm_configure(config: dict = None):
    """
    Configure and initialize a DSPy language model (LM) instance using the provided settings.

    This function sets up the `dspy.LM` object with a specified model name, base URL, and API key.
    It retrieves `base_url` and `api_key` from environment variables (`OPEN_ROUTER_URL` and `OPEN_ROUTER_API_KEY`).

    Args:
        config (dict, optional): A configuration dictionary containing:
            - "model" (str): The model name or identifier (e.g., "openai/gpt-4", "mistralai/mistral-7b").

    Returns:
        dspy.LM: An instance of the DSPy language model configured with the specified parameters.

    Raises:
        KeyError: If the "model" key is missing from the config dictionary.
        EnvironmentError: If required environment variables are not set.
    """
    api_keys = get_config("api_keys")
    lm = dspy.LM(
            model=config["model"],
            base_url=api_keys["OPEN_ROUTER_URL"],
            api_key=api_keys["OPEN_ROUTER_API_KEY"],
            cache=False,
            cache_in_memory=False,
            track_usage=True,
        )
    return lm