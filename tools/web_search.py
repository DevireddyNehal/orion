import os
from abc import ABC, abstractmethod
from dotenv import find_dotenv, load_dotenv
from loguru import logger

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


class BaseSearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 3) -> str:
        """Executes web search and returns formatted snippets."""
        pass


class TavilySearchProvider(BaseSearchProvider):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

    def search(self, query: str, max_results: int = 3) -> str:
        if not self.api_key:
            logger.error("[Tavily] TAVILY_API_KEY is missing from environment.")
            return "Error: TAVILY_API_KEY is not configured."

        try:
            from tavily import TavilyClient
            logger.info(f"[Tavily] Searching web for: {repr(query)}")
            client = TavilyClient(api_key=self.api_key)
            response = client.search(
                query,
                search_depth="basic",
                max_results=max_results,
                include_answer=False  # Disabled slow/hallucinatory AI answer generator for speed & direct accuracy
            )

            output_parts = []
            results = response.get("results", [])
            for i, res in enumerate(results[:max_results], 1):
                title = res.get("title", "No Title")
                snippet = res.get("content", "").strip()
                output_parts.append(f"[{i}] {title}: {snippet}")

            formatted = "\n".join(output_parts)
            return formatted if formatted.strip() else "No relevant search results found."

        except Exception as e:
            logger.error(f"[Tavily Error] {e}")
            return f"Tavily search error: {e}"


class DuckDuckGoSearchProvider(BaseSearchProvider):
    def search(self, query: str, max_results: int = 3) -> str:
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            logger.info(f"[DuckDuckGo] Searching web for: {repr(query)}")
            is_temporal = any(w in query.lower() for w in ("last", "latest", "next", "recent", "today", "yesterday", "winner", "score", "upcoming", "schedule", "won", "current"))

            output_parts = []
            with DDGS() as ddgs:
                if is_temporal:
                    # Try news search first for recent sports and real-time events
                    try:
                        news_results = list(ddgs.news(query, max_results=max_results))
                        for i, res in enumerate(news_results[:max_results], 1):
                            title = res.get("title", "No Title")
                            date = res.get("date", "")
                            snippet = res.get("body", "").strip()
                            date_str = f" ({date[:10]})" if date else ""
                            output_parts.append(f"[{i}] {title}{date_str}: {snippet}")
                    except Exception:
                        pass

                # If news search didn't yield enough, fallback/supplement with web text search
                if len(output_parts) < max_results:
                    remaining = max_results - len(output_parts)
                    timelimit = "m" if is_temporal else None
                    text_results = list(ddgs.text(query, max_results=remaining, timelimit=timelimit))
                    offset = len(output_parts) + 1
                    for i, res in enumerate(text_results[:remaining], offset):
                        title = res.get("title", "No Title")
                        snippet = res.get("body", "").strip()
                        output_parts.append(f"[{i}] {title}: {snippet}")

            formatted = "\n".join(output_parts)
            return formatted if formatted.strip() else "No relevant search results found."

        except Exception as e:
            logger.error(f"[DuckDuckGo Error] {e}")
            return f"DuckDuckGo search error: {e}"


class SerpAPISearchProvider(BaseSearchProvider):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("SERP_API_KEY")

    def search(self, query: str, max_results: int = 3) -> str:
        if not self.api_key:
            logger.error("[SerpAPI] SERP_API_KEY is missing from environment.")
            return "Error: SERP_API_KEY is not configured."

        try:
            import httpx
            logger.info(f"[SerpAPI / Google] Searching web for: {repr(query)}")
            resp = httpx.get(
                "https://serpapi.com/search",
                params={
                    "q": query,
                    "api_key": self.api_key,
                    "engine": "google",
                    "num": max_results
                },
                timeout=10.0
            )
            resp.raise_for_status()
            data = resp.json()

            output_parts = []

            # 1. Answer Box / Knowledge Graph direct answer
            answer_box = data.get("answer_box", {})
            if answer_box:
                ans = answer_box.get("answer") or answer_box.get("snippet") or answer_box.get("result")
                if ans:
                    output_parts.append(f"Google Direct Answer: {ans}")

            # 2. Sports Results
            sports = data.get("sports_results", {})
            if sports:
                game_title = sports.get("title", "")
                tables = sports.get("tables", {})
                output_parts.append(f"Sports Card: {game_title} {tables}")

            # 3. Organic Results
            organic = data.get("organic_results", [])
            for i, res in enumerate(organic[:max_results], 1):
                title = res.get("title", "No Title")
                snippet = res.get("snippet", "").strip()
                output_parts.append(f"[{i}] {title}: {snippet}")

            formatted = "\n".join(output_parts)
            return formatted if formatted.strip() else "No relevant search results found."

        except Exception as e:
            logger.error(f"[SerpAPI Error] {e}")
            return f"SerpAPI search error: {e}"


def get_search_provider(provider_name: str | None = None) -> BaseSearchProvider:
    if not provider_name:
        provider_name = os.getenv("SEARCH_PROVIDER", "duckduckgo").lower().strip()

    if provider_name in ("serpapi", "google", "google_search"):
        return SerpAPISearchProvider()
    elif provider_name in ("duckduckgo", "ddg", "ddgs"):
        return DuckDuckGoSearchProvider()
    elif provider_name in ("tavily", "tavily_search"):
        return TavilySearchProvider()
    else:
        logger.warning(f"Unknown SEARCH_PROVIDER '{provider_name}'. Defaulting to DuckDuckGoSearchProvider.")
        return DuckDuckGoSearchProvider()


def web_search(query: str) -> str:
    """Dispatches search to the configured SEARCH_PROVIDER."""
    provider = get_search_provider()
    return provider.search(query)


if __name__ == "__main__":
    result = web_search("What is the next upcoming Marvel movie?")
    print("Result:\n", result)