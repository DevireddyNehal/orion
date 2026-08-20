from typing import Any, Callable, Dict, List
from loguru import logger
from .web_search import web_search

TOOLS: Dict[str, Callable[..., Any]] = {
    "web_search": web_search,
}

AVAILABLE_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web for recent news, real-time facts, current weather, scores, or information not contained in your offline knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string. Use natural, concise search terms (e.g. 'most recent F1 race winner', 'next upcoming Formula 1 race schedule')."
                    }
                }
            }
        }
    }
]


def execute_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> str:
    """Executes a registered tool by name with the given arguments."""
    if arguments is None:
        arguments = {}

    if tool_name not in TOOLS:
        logger.error(f"[Tool Registry] Unknown tool requested: {tool_name}")
        return f"Error: Unknown tool '{tool_name}'."

    logger.info(f"[Tool Registry] Executing '{tool_name}' with args: {arguments}")
    try:
        if tool_name == "web_search" and not arguments.get("query"):
            return "No search query specified."
        func = TOOLS[tool_name]
        result = func(**arguments)
        return str(result)
    except Exception as e:
        logger.exception(f"[Tool Execution Error] Failed to run '{tool_name}': {e}")
        return f"Tool execution error: {e}"