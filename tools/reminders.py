import re
import uuid
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Callable
from loguru import logger

WORD_TO_NUMBER = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'fifteen': 15, 'twenty': 20,
    'thirty': 30, 'forty-five': 45, 'a': 1, 'an': 1
}


def parse_reminder_delay(time_str: str = "", delay_seconds: Optional[float] = None) -> float:
    """
    Parses natural duration expressions or clock times into total seconds.
    Examples:
      - delay_seconds=300 -> 300.0
      - 'in 5 minutes' / '5 mins' -> 300.0
      - 'in 30 seconds' / '30s' -> 30.0
      - 'in 1 hour' / '1 hr' -> 3600.0
      - '7:30 PM' -> seconds from now until 7:30 PM today (or tomorrow if past)
    """
    if delay_seconds is not None and delay_seconds > 0:
        return delay_seconds

    if not time_str or not time_str.strip():
        raise ValueError("No valid duration or time string provided.")

    text = time_str.strip().lower()

    # 1. Look for relative duration (e.g. 'in 5 minutes', '30 seconds', '2 hours', 'in 10 sec')
    dur_match = re.search(
        r'(?:in\s+)?(\w+)\s*(sec|secs|second|seconds|min|mins|minute|minutes|hr|hrs|hour|hours|h|m|s)\b',
        text
    )
    if dur_match:
        val_str = dur_match.group(1)
        unit = dur_match.group(2)

        if val_str.isdigit():
            amount = float(val_str)
        elif val_str in WORD_TO_NUMBER:
            amount = float(WORD_TO_NUMBER[val_str])
        else:
            amount = 1.0

        if unit in ('sec', 'secs', 'second', 'seconds', 's'):
            return max(1.0, amount)
        elif unit in ('min', 'mins', 'minute', 'minutes', 'm'):
            return max(1.0, amount * 60.0)
        elif unit in ('hr', 'hrs', 'hour', 'hours', 'h'):
            return max(1.0, amount * 3600.0)

    # 2. HH:MM AM/PM or HH:MM clock time
    clock_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text)
    if clock_match:
        hour = int(clock_match.group(1))
        minute = int(clock_match.group(2)) if clock_match.group(2) else 0
        meridiem = clock_match.group(3)

        if meridiem == 'pm' and hour < 12:
            hour += 12
        elif meridiem == 'am' and hour == 12:
            hour = 0

        if 0 <= hour <= 23 and 0 <= minute <= 59:
            now = datetime.now()
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                # If target time already passed today, assume tomorrow
                target += timedelta(days=1)
            diff = (target - now).total_seconds()
            return max(1.0, diff)

    raise ValueError(f"Could not parse duration or time from '{time_str}'")


def format_duration(seconds: float) -> str:
    """Formats seconds into human-readable text (e.g. '5 minutes', '30 seconds', '1 hour 15 minutes')."""
    total_sec = round(seconds)
    if total_sec < 60:
        return f"{total_sec} second{'s' if total_sec != 1 else ''}"
    
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if secs > 0 and hours == 0 and minutes < 5:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")

    return " ".join(parts) if parts else f"{total_sec} seconds"


class ReminderManager:
    def __init__(self):
        self.reminders: Dict[str, Dict[str, Any]] = {}
        self.tts: Any = None
        self._lock = threading.Lock()

    def set_tts(self, tts: Any):
        """Sets the TTS instance used to speak reminders out loud."""
        self.tts = tts

    def set_reminder(self, message: str, time_str: str = "", delay_seconds: Optional[float] = None) -> str:
        """Sets a background internal timer reminder."""
        if not message or not message.strip():
            return "Error: Reminder message cannot be empty."

        clean_message = message.strip()

        try:
            delay = parse_reminder_delay(time_str=time_str, delay_seconds=delay_seconds)
        except Exception as e:
            return f"Failed to set reminder: {e}"

        reminder_id = f"rem_{uuid.uuid4().hex[:6]}"
        now = datetime.now()
        target_time = now + timedelta(seconds=delay)

        timer = threading.Timer(delay, self._on_reminder_expire, args=[reminder_id])
        timer.daemon = True

        with self._lock:
            self.reminders[reminder_id] = {
                "id": reminder_id,
                "message": clean_message,
                "created_at": now,
                "target_time": target_time,
                "delay_seconds": delay,
                "timer": timer,
                "active": True
            }
            timer.start()

        target_str = target_time.strftime("%I:%M:%S %p").lstrip("0")
        dur_str = format_duration(delay)
        logger.info(f"[Reminder Set] ID: {reminder_id} | '{clean_message}' in {dur_str} (Target: {target_str})")
        return f"Reminder set: '{clean_message}' in {dur_str} (at {target_str})."

    def _on_reminder_expire(self, reminder_id: str):
        """Callback fired when internal timer expires."""
        reminder = None
        with self._lock:
            if reminder_id in self.reminders and self.reminders[reminder_id]["active"]:
                reminder = self.reminders[reminder_id]
                reminder["active"] = False

        if not reminder:
            return

        message = reminder["message"]
        logger.info(f"🔔 [REMINDER TRIGGERED] ID: {reminder_id} | Message: '{message}'")

        spoken_text = f"Sir, here is your reminder: {message}."

        if self.tts is not None and hasattr(self.tts, "speak_text"):
            try:
                self.tts.speak_text(spoken_text)
            except Exception as e:
                logger.error(f"[Reminder TTS Error] Failed to speak reminder: {e}")
        else:
            logger.warning(f"[Reminder] TTS engine not connected. Reminder text: '{spoken_text}'")

    def list_reminders(self) -> str:
        """Lists all active pending internal reminders."""
        with self._lock:
            active_list = [r for r in self.reminders.values() if r["active"]]

        if not active_list:
            return "No active internal reminders currently set."

        now = datetime.now()
        lines = ["Active Internal Reminders:"]
        for idx, r in enumerate(active_list, 1):
            rem_id = r["id"]
            msg = r["message"]
            target = r["target_time"]
            remaining = max(0.0, (target - now).total_seconds())
            target_str = target.strftime("%I:%M:%S %p").lstrip("0")
            rem_str = format_duration(remaining)
            lines.append(f"{idx}. [{rem_id}] '{msg}' in {rem_str} (at {target_str})")

        return "\n".join(lines)

    def cancel_reminder(self, identifier: str) -> str:
        """Cancels an active reminder by ID or message keyword match."""
        if not identifier or not identifier.strip():
            return "Error: Please specify a reminder ID or keyword to cancel."

        query = identifier.strip().lower()
        cancelled_item = None

        with self._lock:
            for r_id, r in self.reminders.items():
                if r["active"]:
                    if query == r_id.lower() or query in r["message"].lower():
                        r["active"] = False
                        r["timer"].cancel()
                        cancelled_item = r
                        break

        if cancelled_item:
            msg = cancelled_item["message"]
            rem_id = cancelled_item["id"]
            logger.info(f"[Reminder Cancelled] ID: {rem_id} | '{msg}'")
            return f"Reminder [{rem_id}] '{msg}' has been cancelled."
        else:
            return f"No active reminder matching '{identifier}' was found."


# Global singleton instance
reminder_manager = ReminderManager()


def set_reminder(message: str, time_str: str = "", delay_seconds: Optional[float] = None) -> str:
    """Public tool wrapper to set an internal reminder timer."""
    return reminder_manager.set_reminder(message=message, time_str=time_str, delay_seconds=delay_seconds)


def list_reminders() -> str:
    """Public tool wrapper to list active internal reminders."""
    return reminder_manager.list_reminders()


def cancel_reminder(identifier: str) -> str:
    """Public tool wrapper to cancel an active internal reminder."""
    return reminder_manager.cancel_reminder(identifier=identifier)
