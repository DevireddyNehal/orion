import os

from dotenv import find_dotenv, load_dotenv
from tavily import TavilyClient

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

tavily_api_key = os.getenv("TAVILY_API_KEY")
client = TavilyClient(tavily_api_key)
response = client.search("What are the latest developments in AI this week?")
print(response.items())