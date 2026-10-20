import inspect
from collections.abc import Callable
from typing import Any

from loguru import logger

from .gnome_clock import (
    delete_gnome_alarm,
    list_gnome_alarms,
    set_gnome_alarm,
    set_gnome_timer,
)
from .google_calendar import (
    create_calendar_event,
    delete_calendar_event,
    get_today_agenda,
    list_upcoming_events,
)
from .reminders import (
    cancel_reminder,
    list_reminders,
    set_reminder,
)
from .web_search import web_search

TOOLS: dict[str, Callable[..., Any]] = {
    "web_search": web_search,
    "set_gnome_alarm": set_gnome_alarm,
    "list_gnome_alarms": list_gnome_alarms,
    "delete_gnome_alarm": delete_gnome_alarm,
    "set_gnome_timer": set_gnome_timer,
    "set_reminder": set_reminder,
    "list_reminders": list_reminders,
    "cancel_reminder": cancel_reminder,
    "get_today_agenda": get_today_agenda,
    "list_upcoming_events": list_upcoming_events,
    "create_calendar_event": create_calendar_event,
    "delete_calendar_event": delete_calendar_event,
}

AVAILABLE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web for recent news, real-time facts, current weather, scores, or information not contained in your offline knowledge. Always specify the full search text in 'query'. Pagination, ids, or cursor parameters are not supported.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Optional number of search results to return (default: 3, max: 10).",
                    },
                },
            },
        },
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
                        "description": "The alarm time expression (e.g., '7:30 AM', '19:45', 'in 20 minutes').",
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional label or description for the alarm (e.g., 'Morning Workout', 'Team Sync').",
                    },
                    "sound": {
                        "type": "string",
                        "description": "Alarm sound name. Default is 'sonar'.",
                    },
                },
                "required": ["time_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_gnome_alarms",
            "description": "List all active alarms currently set in Ubuntu's native GNOME Clocks app.",
            "parameters": {"type": "object", "properties": {}},
        },
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
                        "description": "The name or ID of the alarm to remove.",
                    }
                },
                "required": ["name_or_id"],
            },
        },
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
                        "description": "Duration of the timer in seconds (e.g., 300 for 5 minutes).",
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional label for the timer (e.g., 'Tea timer', 'Pasta').",
                    },
                },
                "required": ["duration_seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set an internal voice reminder timer in Orion. When the timer expires, Orion will speak the reminder out loud to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The reminder text or task (e.g., 'check the oven', 'call John', 'take medication').",
                    },
                    "time_str": {
                        "type": "string",
                        "description": "Time duration or expression (e.g., '5 minutes', '30 seconds', 'in 1 hour', '7:30 PM').",
                    },
                    "delay_seconds": {
                        "type": "number",
                        "description": "Optional direct delay in seconds (e.g. 300 for 5 minutes).",
                    },
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "List all active pending internal reminders set in Orion.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reminder",
            "description": "Cancel an active pending internal reminder in Orion by ID or keyword match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "The reminder ID or keyword matching the reminder message to cancel.",
                    }
                },
                "required": ["identifier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_today_agenda",
            "description": "Fetch all Google Calendar events scheduled for today in the user's primary calendar.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_upcoming_events",
            "description": "List upcoming events from the user's primary Google Calendar over the next N days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of events to return (default: 15).",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days ahead to search (default: 30).",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional search term to filter events by keyword or topic (e.g. 'hackathon', 'meeting', 'flight', 'doctor').",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create and schedule a new event on the user's primary Google Calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Title or summary of the meeting/event.",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Start time expression (e.g. 'today at 3pm', 'tomorrow at 10:30 am', 'in 2 hours', '2026-10-21 15:00').",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "Optional end time expression. Defaults to 1 hour after start time if omitted.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional notes or description for the event.",
                    },
                    "location": {
                        "type": "string",
                        "description": "Optional event location or video call link.",
                    },
                },
                "required": ["summary", "start_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_calendar_event",
            "description": "Delete a calendar event from Google Calendar by ID or exact title match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id_or_summary": {
                        "type": "string",
                        "description": "The event ID or title of the event to delete.",
                    }
                },
                "required": ["event_id_or_summary"],
            },
        },
    },
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
        if tool_name == "web_search" and not str(arguments.get("query") or "").strip():
            return "Error: No search query specified. You must provide a full search query string in 'query'. Pagination, ids, and cursors are not supported."
        func = TOOLS[tool_name]
        sig = inspect.signature(func)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        if has_var_keyword:
            result = func(**arguments)
        else:
            filtered_args = {k: v for k, v in arguments.items() if k in sig.parameters}
            result = func(**filtered_args)

        return str(result)
    except Exception as e:
        logger.exception(f"[Tool Execution Error] Failed to run '{tool_name}': {e}")
        return f"Tool execution error: {e}"
