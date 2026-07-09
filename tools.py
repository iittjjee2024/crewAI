import os
from crewai.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper

# Ensure the Google Search wrapper uses the environment variables we provide
# It automatically looks for GOOGLE_API_KEY and GOOGLE_CSE_ID in the environment.
try:
    search = GoogleSearchAPIWrapper()
except Exception as e:
    search = None
    print(f"Warning: Failed to initialize Google Search API: {e}")
    print("Please ensure GOOGLE_API_KEY and GOOGLE_CSE_ID are set in your .env file.")

@tool("Google Custom Search")
def google_search_tool(query: str) -> str:
    """
    Search Google for the given query and return the top results.
    Useful for researching factual information, mythological lore, and historical context.
    """
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
            formatted_results.append(f"{i+1}. {title}\\n   Summary: {snippet}\\n   Source: {link}\\n")
            
        return "\\n".join(formatted_results)
    except Exception as e:
        return f"Failed to execute search. Error: {str(e)}"
