from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

# OAuth & Google Calendar API libraries
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from loguru import logger

SCOPES = ["https://www.googleapis.com/auth/calendar"]

CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH", "token.json")


def get_local_timezone() -> ZoneInfo:
    """Returns local system IANA timezone (e.g. Asia/Kolkata) or defaults to UTC."""
    # 1. Explicit environment variable override
    tz_env = os.getenv("TZ")
    if tz_env:
        try:
            return ZoneInfo(tz_env)
        except Exception:
            pass

    # 2. Linux canonical timezone from /etc/localtime symlink
    if os.path.exists("/etc/localtime"):
        try:
            real_path = os.path.realpath("/etc/localtime")
            if "zoneinfo/" in real_path:
                iana_name = real_path.split("zoneinfo/", 1)[1]
                return ZoneInfo(iana_name)
        except Exception:
            pass

    # 3. Linux /etc/timezone file (Debian/Ubuntu)
    if os.path.exists("/etc/timezone"):
        try:
            with open("/etc/timezone") as f:
                name = f.read().strip()
                if name:
                    return ZoneInfo(name)
        except Exception:
            pass

    return ZoneInfo("UTC")


def get_calendar_service() -> Any:
    """
    Returns an authenticated Google Calendar API service instance.
    Loads from token.json if present; refreshes expired tokens;
    or initiates a one-time desktop browser OAuth flow if needed.
    """
    creds: Any = None

    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        except Exception as e:
            logger.warning(f"[GoogleCalendar] Failed to load {TOKEN_PATH}: {e}")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                logger.info("[GoogleCalendar] Refreshing expired OAuth token...")
                creds.refresh(Request())
                with open(TOKEN_PATH, "w") as token_file:
                    token_file.write(creds.to_json())
                logger.info("[GoogleCalendar] Token refreshed successfully.")
            except Exception as e:
                logger.error(f"[GoogleCalendar] Failed to refresh token: {e}")
                creds = None

        if not creds:
            if not os.path.exists(CREDENTIALS_PATH):
                logger.error(
                    f"[GoogleCalendar] '{CREDENTIALS_PATH}' not found. "
                    "Please download your OAuth client ID JSON from Google Cloud Console."
                )
                return None

            try:
                logger.info(
                    "[GoogleCalendar] Initiating one-time browser OAuth authentication..."
                )
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_PATH, SCOPES
                )
                creds = flow.run_local_server(port=0)
                if creds:
                    with open(TOKEN_PATH, "w") as token_file:
                        token_file.write(creds.to_json())
                    logger.info(
                        f"[GoogleCalendar] Authorization successful. Saved token to {TOKEN_PATH}."
                    )
            except Exception as e:
                logger.error(f"[GoogleCalendar] OAuth authorization failed: {e}")
                return None

    try:
        return build("calendar", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        logger.error(f"[GoogleCalendar] Failed to build Google Calendar service: {e}")
        return None


def parse_calendar_datetime(
    time_str: str, default_duration_minutes: int = 60
) -> tuple[datetime, datetime]:
    """
    Parses flexible human expressions (e.g. 'today at 3pm', 'tomorrow at 10:30 am',
    'in 2 hours', '2026-10-21 15:00') into (start_dt, end_dt) with local timezone.
    """
    tz = get_local_timezone()
    now = datetime.now(tz)
    raw_str = time_str.strip().lower()

    # 1. Direct ISO format check (e.g. from LLM: '2026-10-20T21:00:00' or '2026-10-20 21:00')
    iso_candidate = raw_str.replace("z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso_candidate)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        else:
            dt = dt.astimezone(tz)
        return dt, dt + timedelta(minutes=default_duration_minutes)
    except Exception:
        pass

    # 2. Pattern: "in X hours" / "in X minutes"
    rel_match = re.search(r"in\s+(\d+)\s*(hour|hr|minute|min)s?", raw_str)
    if rel_match:
        qty = int(rel_match.group(1))
        unit = rel_match.group(2)
        delta = timedelta(hours=qty) if "h" in unit else timedelta(minutes=qty)
        start_dt = now + delta
        end_dt = start_dt + timedelta(minutes=default_duration_minutes)
        return start_dt, end_dt

    is_explicit_tomorrow = "tomorrow" in raw_str
    is_explicit_today = "today" in raw_str

    # Determine base date: today, tomorrow, or a specific day
    base_date = now.date()
    if is_explicit_tomorrow:
        base_date = base_date + timedelta(days=1)
        raw_str = raw_str.replace("tomorrow", "").strip()
    elif is_explicit_today:
        raw_str = raw_str.replace("today", "").strip()

    # Look for time of day: e.g. "3pm", "15:00", "10:30 am", "noon", "9 pm", "9:00"
    hour = 9
    minute = 0

    if "noon" in raw_str:
        hour, minute = 12, 0
    elif "midnight" in raw_str:
        hour, minute = 0, 0
    else:
        time_match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", raw_str)
        if time_match:
            raw_h = int(time_match.group(1))
            raw_m = int(time_match.group(2)) if time_match.group(2) else 0
            meridiem = time_match.group(3)

            if meridiem:
                if meridiem == "pm" and raw_h < 12:
                    raw_h += 12
                elif meridiem == "am" and raw_h == 12:
                    raw_h = 0
            hour, minute = raw_h, raw_m

    start_dt = datetime(
        base_date.year,
        base_date.month,
        base_date.day,
        hour,
        minute,
        tzinfo=tz,
    )
    # If neither today nor tomorrow was explicitly stated, and time already passed today, assume tomorrow
    if not is_explicit_today and not is_explicit_tomorrow and start_dt < now:
        start_dt += timedelta(days=1)

    end_dt = start_dt + timedelta(minutes=default_duration_minutes)
    return start_dt, end_dt


def get_today_agenda() -> list[dict[str, Any]]:
    """
    Fetches all calendar events scheduled for today in the user's primary calendar.
    """
    service = get_calendar_service()
    if not service:
        return []

    tz = get_local_timezone()
    now = datetime.now(tz)
    start_of_day = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=tz)
    end_of_day = datetime(now.year, now.month, now.day, 23, 59, 59, tzinfo=tz)

    try:
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=start_of_day.isoformat(),
                timeMax=end_of_day.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        raw_events = events_result.get("items", [])
        return _simplify_events(raw_events)
    except Exception as e:
        logger.error(f"[GoogleCalendar] Failed to fetch today's agenda: {e}")
        return []


def list_upcoming_events(
    max_results: int = 15, days: int = 30, query: str | None = None
) -> list[dict[str, Any]]:
    """
    Lists upcoming events from the user's primary Google Calendar over the next N days.
    Optionally filters events by search query text (e.g. 'hackathon', 'meeting', 'interview').
    """
    service = get_calendar_service()
    if not service:
        return []

    tz = get_local_timezone()
    now = datetime.now(tz)
    time_max = (now + timedelta(days=days)).isoformat()

    try:
        kwargs: dict[str, Any] = {
            "calendarId": "primary",
            "timeMin": now.isoformat(),
            "timeMax": time_max,
            "maxResults": max_results,
            "singleEvents": True,
            "orderBy": "startTime",
        }
        if query and query.strip():
            kwargs["q"] = query.strip()

        events_result = service.events().list(**kwargs).execute()
        raw_events = events_result.get("items", [])
        return _simplify_events(raw_events)
    except Exception as e:
        logger.error(f"[GoogleCalendar] Failed to list upcoming events: {e}")
        return []


def create_calendar_event(
    summary: str,
    start_time: str,
    end_time: str | None = None,
    description: str = "",
    location: str = "",
) -> dict[str, Any]:
    """
    Creates and schedules a new event on the user's primary Google Calendar.
    """
    service = get_calendar_service()
    if not service:
        return {"status": "error", "message": "Google Calendar service not available"}

    start_dt, calc_end_dt = parse_calendar_datetime(start_time)
    if end_time:
        end_dt, _ = parse_calendar_datetime(end_time)
        if end_dt <= start_dt:
            end_dt = start_dt + timedelta(minutes=60)
    else:
        end_dt = calc_end_dt

    event_body = {
        "summary": summary,
        "description": description,
        "location": location,
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": str(start_dt.tzinfo),
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": str(end_dt.tzinfo),
        },
    }

    try:
        created_event = (
            service.events()
            .insert(calendarId="primary", body=event_body)
            .execute()
        )
        logger.info(
            f"[GoogleCalendar] Created event '{summary}' (ID: {created_event.get('id')}) at {start_dt.isoformat()}"
        )
        return {
            "status": "success",
            "id": created_event.get("id"),
            "summary": summary,
            "start": start_dt.strftime("%A, %b %d at %I:%M %p"),
            "end": end_dt.strftime("%I:%M %p"),
            "link": created_event.get("htmlLink", ""),
        }
    except Exception as e:
        logger.error(f"[GoogleCalendar] Failed to create event '{summary}': {e}")
        return {"status": "error", "message": str(e)}


def delete_calendar_event(event_id_or_summary: str) -> dict[str, Any]:
    """
    Deletes an event from Google Calendar by ID or exact summary match.
    """
    service = get_calendar_service()
    if not service:
        return {"status": "error", "message": "Google Calendar service not available"}

    target_id = event_id_or_summary.strip()

    # If user provided a title/summary rather than an event ID, search upcoming events for match
    if not re.match(r"^[a-zA-Z0-9_]{10,}$", target_id):
        upcoming = list_upcoming_events(max_results=20, days=14)
        for ev in upcoming:
            if ev["summary"].lower() == target_id.lower():
                target_id = ev["id"]
                break

    try:
        service.events().delete(calendarId="primary", eventId=target_id).execute()
        logger.info(f"[GoogleCalendar] Deleted event ID: {target_id}")
        return {"status": "success", "deleted_id": target_id}
    except Exception as e:
        logger.error(f"[GoogleCalendar] Failed to delete event '{target_id}': {e}")
        return {"status": "error", "message": str(e)}


def format_events_for_speech(
    events: list[dict[str, Any]], context_title: str = "today"
) -> str:
    """
    Formats a list of calendar events into crisp, natural spoken English for Kokoro TTS.
    """
    if not events:
        if context_title == "today":
            return "Your calendar is completely clear today, sir."
        return f"You have no upcoming events scheduled for {context_title}, sir."

    count = len(events)
    if count == 1:
        ev = events[0]
        time_part = ev["formatted_time"]
        return f"You have one event on your calendar {context_title}, sir: {ev['summary']} at {time_part}."

    items_spoken = []
    for ev in events[:4]:  # Limit to 4 for voice prosody
        items_spoken.append(f"{ev['summary']} at {ev['formatted_time']}")

    joined = ", and ".join(items_spoken) if len(items_spoken) == 2 else ", ".join(items_spoken[:-1]) + f", and {items_spoken[-1]}"
    return f"You have {count} events {context_title}, sir: {joined}."


def _simplify_events(raw_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Helper to convert Google Calendar API raw items into clean dicts."""
    simplified = []
    for item in raw_events:
        start_obj = item.get("start", {})
        start_raw = start_obj.get("dateTime") or start_obj.get("date", "")
        end_obj = item.get("end", {})
        end_raw = end_obj.get("dateTime") or end_obj.get("date", "")

        # Format human-readable time string for speech
        formatted_time = start_raw
        try:
            if "t" in start_raw.lower():
                dt = datetime.fromisoformat(start_raw)
                formatted_time = dt.strftime("%-I:%M %p")
            else:
                formatted_time = "all day"
        except Exception:
            pass

        # Truncate description to prevent large payloads from inflating LLM latency
        raw_desc = item.get("description", "") or ""
        clean_desc = raw_desc[:250].strip() if raw_desc else ""

        simplified.append(
            {
                "id": item.get("id"),
                "summary": item.get("summary", "Untitled event"),
                "start": start_raw,
                "end": end_raw,
                "formatted_time": formatted_time,
                "location": item.get("location", ""),
                "description": clean_desc,
            }
        )
    return simplified
