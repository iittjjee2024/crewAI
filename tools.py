import os
from dotenv import load_dotenv
from crewai.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper

# Load .env variables so the Google Search wrapper can pick them up
load_dotenv()

# Lazy-init: the wrapper is created inside the tool function to ensure
# env vars are loaded before the wrapper reads them.
_search_wrapper = None

def _get_search():
    """Return a (possibly cached) GoogleSearchAPIWrapper instance."""
    global _search_wrapper
    if _search_wrapper is None:
        try:
            _search_wrapper = GoogleSearchAPIWrapper()
        except Exception as e:
            print(f"Warning: Failed to initialize Google Search API: {e}")
            print("Please ensure GOOGLE_API_KEY and GOOGLE_CSE_ID are set in your .env file.")
            _search_wrapper = None
    return _search_wrapper

@tool("Google Custom Search")
def google_search_tool(query: str) -> str:
    """
    Search Google for the given query and return the top results.
    Useful for researching factual information, mythological lore, and historical context.
    """
    search = _get_search()
    if search is None:
        return "Error: Google Search API is not configured properly."

    try:
        # Get the top 5 results
        results = search.results(query, 5)

        # Format the results into a readable string
        formatted_results = []
        for i, res in enumerate(results):
            title = res.get("title", "No Title")
            snippet = res.get("snippet", "No Snippet")
            link = res.get("link", "No Link")
            formatted_results.append(f"{i+1}. {title}\n   Summary: {snippet}\n   Source: {link}\n")

        return "\n".join(formatted_results)
    except Exception as e:
        return f"Failed to execute search. Error: {str(e)}"
