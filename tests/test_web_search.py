import unittest
from unittest.mock import MagicMock, patch

from tools.web_search import (
    DuckDuckGoSearchProvider,
    SerpAPISearchProvider,
    TavilySearchProvider,
    get_search_provider,
    web_search,
)


class TestWebSearch(unittest.TestCase):
    def test_get_search_provider_resolution(self):
        # DuckDuckGo aliases
        self.assertIsInstance(
            get_search_provider("duckduckgo"), DuckDuckGoSearchProvider
        )
        self.assertIsInstance(get_search_provider("ddg"), DuckDuckGoSearchProvider)
        self.assertIsInstance(get_search_provider("ddgs"), DuckDuckGoSearchProvider)

        # Tavily aliases
        self.assertIsInstance(get_search_provider("tavily"), TavilySearchProvider)
        self.assertIsInstance(
            get_search_provider("tavily_search"), TavilySearchProvider
        )

        # SerpAPI aliases
        self.assertIsInstance(get_search_provider("serpapi"), SerpAPISearchProvider)
        self.assertIsInstance(get_search_provider("google"), SerpAPISearchProvider)

        # Unknown provider fallback
        self.assertIsInstance(
            get_search_provider("unknown_engine"), DuckDuckGoSearchProvider
        )

    def test_get_search_provider_default_from_env(self):
        with patch.dict("os.environ", {"SEARCH_PROVIDER": "tavily"}):
            provider = get_search_provider()
            self.assertIsInstance(provider, TavilySearchProvider)

        with patch.dict("os.environ", {"SEARCH_PROVIDER": "duckduckgo"}):
            provider = get_search_provider()
            self.assertIsInstance(provider, DuckDuckGoSearchProvider)

    def test_web_search_argument_aliases(self):
        mock_provider = MagicMock()
        mock_provider.search.return_value = "Mock Search Result"

        with patch("tools.web_search.get_search_provider", return_value=mock_provider):
            # Test alias: top_n
            web_search("quantum computing", top_n=5)
            mock_provider.search.assert_called_with("quantum computing", max_results=5)

            # Test alias: limit
            web_search("latest news", limit=4)
            mock_provider.search.assert_called_with("latest news", max_results=4)

            # Test alias: n
            web_search("weather", n=2)
            mock_provider.search.assert_called_with("weather", max_results=2)

    def test_web_search_bounds_clamping(self):
        mock_provider = MagicMock()
        mock_provider.search.return_value = "Mock Search Result"

        with patch("tools.web_search.get_search_provider", return_value=mock_provider):
            # Clamping lower bound (< 1 -> 1)
            web_search("test query", max_results=-5)
            mock_provider.search.assert_called_with("test query", max_results=1)

            # Clamping upper bound (> 10 -> 10)
            web_search("test query", max_results=50)
            mock_provider.search.assert_called_with("test query", max_results=10)


if __name__ == "__main__":
    unittest.main()
