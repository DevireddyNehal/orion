import os

from dotenv import find_dotenv, load_dotenv
from tavily import TavilyClient

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
tavily_api_key = os.getenv("TAVILY_API_KEY")
def web_search(query: str) -> str:
    client = TavilyClient(tavily_api_key)
    response = client.search(
        query,
        search_depth="basic",
        max_results=3,
        include_answer="basic"
    )
    return response.items()

x=web_search("What is the next marvel movie")
print(x)