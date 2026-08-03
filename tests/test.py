import os

from dotenv import find_dotenv, load_dotenv
from tavily import TavilyClient

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

tavily_api_key = os.getenv("TAVILY_API_KEY")
client = TavilyClient(tavily_api_key)
response = client.search(
    query="When is the next F1 race and who is most likely to win it?",
    topic="news"
)
print(response["results"])