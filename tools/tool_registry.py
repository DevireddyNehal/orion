from typing import Any, Callable, Dict, List
from loguru import logger
from .web_search import web_search
from .gnome_clock import (
    set_gnome_alarm,
    list_gnome_alarms,
    delete_gnome_alarm,
    set_gnome_timer,
)

TOOLS: Dict[str, Callable[..., Any]] = {
    "web_search": web_search,
    "set_gnome_alarm": set_gnome_alarm,
    "list_gnome_alarms": list_gnome_alarms,
    "delete_gnome_alarm": delete_gnome_alarm,
    "set_gnome_timer": set_gnome_timer,
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
                        "description": "The search query string."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_gnome_alarm",
            "description": "Set a new alarm in Ubuntu's native GNOME Clocks app with custom time, label, and Sonar sound.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_str": {
                        "type": "string",
                        "description": "The alarm time expression (e.g., '7:30 AM', '19:45', 'in 20 minutes')."
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional label or description for the alarm (e.g., 'Morning Workout', 'Team Sync')."
                    },
                    "sound": {
                        "type": "string",
                        "description": "Alarm sound name. Default is 'sonar'."
                    }
                },
                "required": ["time_str"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_gnome_alarms",
            "description": "List all active alarms currently set in Ubuntu's native GNOME Clocks app.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_gnome_alarm",
            "description": "Delete an existing alarm in Ubuntu's native GNOME Clocks app by name or ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_or_id": {
                        "type": "string",
                        "description": "The name or ID of the alarm to remove."
                    }
                },
                "required": ["name_or_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_gnome_timer",
            "description": "Start a countdown timer in Ubuntu's native GNOME Clocks app.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_seconds": {
                        "type": "integer",
                        "description": "Duration of the timer in seconds (e.g., 300 for 5 minutes)."
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional label for the timer (e.g., 'Tea timer', 'Pasta')."
                    }
                },
                "required": ["duration_seconds"]
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