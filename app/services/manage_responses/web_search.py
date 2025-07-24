from tavily import AsyncTavilyClient
from dotenv import load_dotenv
import os
# -------------------- Web Search Service Functions --------------------
load_dotenv()
tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

async def web_search(query: str, manual_trigger: bool = False):
    """    
    This function sanitizes the query, performs the search, and formats the results.
    Args:
        query (str): The search query.
        manual_trigger (bool): Whether this search was manually triggered (for logging).
    Returns:
        list: A list of formatted search results.
    Raises:
        ValueError: If the query is empty or exceeds length constraints.
    """
    sanitized_query = sanitize_query(query)
    search_results = await tavily_client.search(
        query=sanitized_query,
        search_depth="advanced",
        max_results=5
    )
    formatted_results = [
        {
            "content": result["content"],
            "url": result["url"],
            "title": result["title"],
            "score": result["score"],
            "external_flag": True
        }
        for result in search_results["results"]
    ]
    log_external_search(sanitized_query, formatted_results, manual_trigger)
    return formatted_results

def sanitize_query(query: str) -> str:
    """
    Sanitize the search query to ensure it meets length and format requirements.
    Args:
        query (str): The raw search query.
    Returns:
        str: A sanitized version of the query.
    Raises:
        ValueError: If the query is empty or exceeds length constraints.
    """
    # Strip leading/trailing whitespace and remove control chars
    query = query.strip().replace("\n", " ").replace("\r", " ")

    # Enforce length constraints
    if len(query) < 1:
        raise ValueError("Query must be at least 1 character long")
    if len(query) > 400:
        query = query[:400]

    return query

def log_external_search(query: str, results: list, manual_trigger: bool):
    """
    Log the details of the external search for debugging and analytics.
    Args:
        query (str): The sanitized search query.
        results (list): The list of search results.
        manual_trigger (bool): Whether this search was manually triggered.
    """
    print(f"[WebSearch] Query='{query}', Manual={manual_trigger}, Results={len(results)}")


# -------------------- Formatter --------------------
def format_search_results(results: list) -> str:
    """
    Format the search results into a human-readable string.
    Args:
        results (list): A list of search result dictionaries.
    Returns:
        str: A formatted string containing the search results.
    """
    return "\n\n".join(
        f"{i+1}. [{r['title']}]({r['url']})\n{r['content'][:200]}..."
        for i, r in enumerate(results)
    )




