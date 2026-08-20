import ast
import json
import re
import subprocess
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from loguru import logger

try:
    import gi  # type: ignore
    gi.require_version('Gio', '2.0')
    from gi.repository import Gio, GLib  # type: ignore
    GSETTINGS_GIO_AVAILABLE = True
except Exception:
    GSETTINGS_GIO_AVAILABLE = False


WORD_TO_NUMBER = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'fifteen': 15, 'twenty': 20,
    'thirty': 30, 'forty-five': 45, 'a': 1, 'an': 1
}

SONAR_SOUND_FILE = "/usr/share/sounds/gnome/default/alarms/sonar.oga"
SONAR_SOUND_URI = "file:///usr/share/sounds/gnome/default/alarms/sonar.oga"


def parse_time(time_str: str) -> tuple[int, int]:
    """
    Parses natural time expressions into (hour_24, minute).
    Examples: '7:30 AM' -> (7, 30), '19:45' -> (19, 45), 'in 1 minute' -> (current+1), 'in one min' -> (current+1).
    """
    time_str = time_str.strip().lower()

    # Relative time: 'in X minutes' or 'in X hours' or 'in one minute'
    rel_match = re.search(r'in\s+(\w+)\s*(min|minute|minutes|hr|hour|hours)', time_str)
    if rel_match:
        val_str = rel_match.group(1)
        unit = rel_match.group(2)

        if val_str.isdigit():
            amount = int(val_str)
        elif val_str in WORD_TO_NUMBER:
            amount = WORD_TO_NUMBER[val_str]
        else:
            amount = 1

        now = datetime.now()
        if 'hr' in unit or 'hour' in unit:
            target = now + timedelta(hours=amount)
        else:
            target = now + timedelta(minutes=amount)
        return target.hour, target.minute

    # HH:MM AM/PM or HH:MM
    match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', time_str)
    if not match:
        raise ValueError(f"Could not parse time string: '{time_str}'")

    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    meridiem = match.group(3)

    if meridiem == 'pm' and hour < 12:
        hour += 12
    elif meridiem == 'am' and hour == 12:
        hour = 0

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid time bounds: {hour:02d}:{minute:02d}")

    return hour, minute


def _refresh_gnome_clocks_daemon():
    """
    Kills any cached background GNOME Clocks GApplication daemon so that
    GNOME Clocks reloads the fresh GSettings payload from disk into the GUI.
    """
    try:
        subprocess.run(["pkill", "-f", "gnome-clocks"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# --- GSettings CLI Helpers ---

def _cli_get_alarms() -> List[Dict[str, Any]]:
    try:
        res = subprocess.run(
            ["gsettings", "get", "org.gnome.clocks", "alarms"],
            capture_output=True, text=True, check=True
        )
        out = res.stdout.strip()
        if out.startswith("@aa{sv}"):
            out = out[7:].strip()
        if not out or out == "[]":
            return []

        cleaned = re.sub(r"<'([^']*)'>", r"'\1'", out)
        cleaned = re.sub(r"<(true|false)>", lambda m: m.group(1).capitalize(), cleaned)
        cleaned = re.sub(r"<(?:uint32|int32|int)?\s*(\d+)>", r"\1", cleaned)
        cleaned = re.sub(r"<@a[iu]\s+\[\]>", "[]", cleaned)

        parsed = ast.literal_eval(cleaned)
        return parsed if isinstance(parsed, list) else []
    except Exception as e:
        logger.warning(f"[GNOME Clock] CLI get alarms error: {e}")
        return []


def _cli_set_alarms(alarms: List[Dict[str, Any]]) -> bool:
    try:
        formatted_items = []
        for a in alarms:
            name = a.get("name", "Alarm")
            hour = int(a.get("hour", 0))
            minute = int(a.get("minute", 0))
            alarm_id = a.get("id", a.get("alarm-id", f"orion_{uuid.uuid4().hex[:8]}"))
            active = "true" if a.get("active", True) else "false"
            snooze_mins = int(a.get("snooze_minutes", 10))
            ring_mins = int(a.get("ring_minutes", 5))

            item_str = (
                f"{{'name': <'{name}'>, 'id': <'{alarm_id}'>, 'hour': <{hour}>, 'minute': <{minute}>, "
                f"'active': <{active}>, 'days': <@ai []>, 'snooze_minutes': <{snooze_mins}>, "
                f"'ring_minutes': <{ring_mins}>, 'sound': <'sonar'>, 'sound_id': <'sonar'>, "
                f"'sound_file': <'{SONAR_SOUND_FILE}'>, 'sound_uri': <'{SONAR_SOUND_URI}'>}}"
            )
            formatted_items.append(item_str)

        val_str = "[" + ", ".join(formatted_items) + "]"
        subprocess.run(
            ["gsettings", "set", "org.gnome.clocks", "alarms", val_str],
            check=True
        )
        _refresh_gnome_clocks_daemon()
        return True
    except Exception as e:
        logger.error(f"[GNOME Clock] CLI set alarms error: {e}")
        return False


def _cli_get_timers() -> List[Dict[str, Any]]:
    try:
        res = subprocess.run(
            ["gsettings", "get", "org.gnome.clocks", "timers"],
            capture_output=True, text=True, check=True
        )
        out = res.stdout.strip()
        if out.startswith("@aa{sv}"):
            out = out[7:].strip()
        if not out or out == "[]":
            return []

        cleaned = re.sub(r"<'([^']*)'>", r"'\1'", out)
        cleaned = re.sub(r"<(true|false)>", lambda m: m.group(1).capitalize(), cleaned)
        cleaned = re.sub(r"<(?:uint32|int32|int\s+)?([0-9.]+)>", r"\1", cleaned)

        parsed = ast.literal_eval(cleaned)
        return parsed if isinstance(parsed, list) else []
    except Exception as e:
        logger.warning(f"[GNOME Clock] CLI get timers error: {e}")
        return []


def _cli_set_timers(timers: List[Dict[str, Any]]) -> bool:
    try:
        formatted_items = []
        for t in timers:
            name = t.get("name", "Timer")
            duration = float(t.get("duration", 0))
            timer_id = t.get("timer-id", f"orion_timer_{uuid.uuid4().hex[:8]}")
            state = t.get("state", 1)

            item_str = (
                f"{{'name': <'{name}'>, 'duration': <{duration}>, "
                f"'timer-id': <'{timer_id}'>, 'state': <{state}>}}"
            )
            formatted_items.append(item_str)

        val_str = "[" + ", ".join(formatted_items) + "]"
        subprocess.run(
            ["gsettings", "set", "org.gnome.clocks", "timers", val_str],
            check=True
        )
        _refresh_gnome_clocks_daemon()
        return True
    except Exception as e:
        logger.error(f"[GNOME Clock] CLI set timers error: {e}")
        return False


# --- Public API Tools ---

def set_gnome_alarm(time_str: str, name: str = "Alarm", sound: str = "sonar") -> str:
    """
    Sets a new alarm in native GNOME Clocks with specified time and sound (default: Sonar).
    """
    try:
        hour, minute = parse_time(time_str)
    except Exception as e:
        return f"Failed to set alarm: {e}"

    alarm_name = name.strip() if name else f"Alarm {hour:02d}:{minute:02d}"
    alarm_id = f"orion_{uuid.uuid4().hex[:8]}"

    if GSETTINGS_GIO_AVAILABLE:
        try:
            settings = Gio.Settings(schema_id="org.gnome.clocks")
            existing_variant = settings.get_value("alarms")
            existing_alarms: List[Dict[str, Any]] = existing_variant.unpack() if existing_variant else []

            new_alarm = {
                'name': GLib.Variant('s', alarm_name),
                'id': GLib.Variant('s', alarm_id),
                'hour': GLib.Variant('i', hour),
                'minute': GLib.Variant('i', minute),
                'active': GLib.Variant('b', True),
                'days': GLib.Variant('ai', []),
                'snooze_minutes': GLib.Variant('i', 10),
                'ring_minutes': GLib.Variant('i', 5),
                'sound': GLib.Variant('s', sound),
                'sound_id': GLib.Variant('s', sound),
                'sound_file': GLib.Variant('s', SONAR_SOUND_FILE),
                'sound_uri': GLib.Variant('s', SONAR_SOUND_URI),
            }

            packed_alarms = []
            for item in existing_alarms:
                item_dict = {}
                for k, v in item.items():
                    if k in ('hour', 'minute', 'snooze_minutes', 'ring_minutes'):
                        item_dict[k] = GLib.Variant('i', int(v))
                    elif isinstance(v, str):
                        item_dict[k] = GLib.Variant('s', v)
                    elif isinstance(v, bool):
                        item_dict[k] = GLib.Variant('b', v)
                    elif isinstance(v, list):
                        item_dict[k] = GLib.Variant('ai', v)
                    elif isinstance(v, int):
                        item_dict[k] = GLib.Variant('i', v)
                packed_alarms.append(item_dict)

            packed_alarms.append(new_alarm)

            new_variant = GLib.Variant('aa{sv}', packed_alarms)
            settings.set_value("alarms", new_variant)
            _refresh_gnome_clocks_daemon()
            formatted_time = f"{hour:02d}:{minute:02d}"
            logger.info(f"[GNOME Clock] Set alarm '{alarm_name}' for {formatted_time} with sound '{sound}' via GIO.")
            return f"Alarm '{alarm_name}' set for {formatted_time} in GNOME Clocks."
        except Exception as e:
            logger.warning(f"[GNOME Clock] GIO set failed ({e}), falling back to CLI...")

    # CLI Subprocess Fallback
    alarms = _cli_get_alarms()
    alarms.append({
        "name": alarm_name,
        "id": alarm_id,
        "hour": hour,
        "minute": minute,
        "sound": sound,
        "sound_id": sound,
        "sound_file": SONAR_SOUND_FILE,
        "sound_uri": SONAR_SOUND_URI,
        "active": True,
        "snooze_minutes": 10,
        "ring_minutes": 5
    })

    if _cli_set_alarms(alarms):
        formatted_time = f"{hour:02d}:{minute:02d}"
        logger.info(f"[GNOME Clock] Set alarm '{alarm_name}' for {formatted_time} with sound '{sound}' via CLI.")
        return f"Alarm '{alarm_name}' set for {formatted_time} in GNOME Clocks."
    else:
        return "Error updating GNOME Clocks settings."


def list_gnome_alarms() -> str:
    """
    Lists all active alarms in native GNOME Clocks.
    """
    alarms: List[Dict[str, Any]] = []

    if GSETTINGS_GIO_AVAILABLE:
        try:
            settings = Gio.Settings(schema_id="org.gnome.clocks")
            alarms_variant = settings.get_value("alarms")
            alarms = alarms_variant.unpack() if alarms_variant else []
        except Exception:
            alarms = _cli_get_alarms()
    else:
        alarms = _cli_get_alarms()

    if not alarms:
        return "No alarms currently set in GNOME Clocks."

    lines = ["Current GNOME Clocks Alarms:"]
    for idx, a in enumerate(alarms, 1):
        name = a.get("name", "Unnamed")
        hour = a.get("hour", 0)
        minute = a.get("minute", 0)
        active = a.get("active", True)
        sound_uri = a.get("sound_uri", a.get("sound", "default"))
        sound_name = "Sonar" if "sonar" in str(sound_uri).lower() else "Default"
        status = "Active" if active else "Disabled"
        lines.append(f"{idx}. '{name}' at {hour:02d}:{minute:02d} ({status}, Sound: {sound_name})")

    return "\n".join(lines)


def delete_gnome_alarm(name_or_id: str) -> str:
    """
    Deletes an alarm from GNOME Clocks by name or ID.
    """
    target = name_or_id.strip().lower()

    if GSETTINGS_GIO_AVAILABLE:
        try:
            settings = Gio.Settings(schema_id="org.gnome.clocks")
            alarms_variant = settings.get_value("alarms")
            alarms = alarms_variant.unpack() if alarms_variant else []

            remaining_packed = []
            found = False

            for a in alarms:
                a_name = str(a.get("name", "")).lower()
                a_id = str(a.get("id", a.get("alarm-id", ""))).lower()
                if target == a_name or target == a_id or target in a_name:
                    found = True
                    continue

                item_dict = {}
                for k, v in a.items():
                    if k in ('hour', 'minute', 'snooze_minutes', 'ring_minutes'):
                        item_dict[k] = GLib.Variant('i', int(v))
                    elif isinstance(v, str):
                        item_dict[k] = GLib.Variant('s', v)
                    elif isinstance(v, bool):
                        item_dict[k] = GLib.Variant('b', v)
                    elif isinstance(v, list):
                        item_dict[k] = GLib.Variant('ai', v)
                    elif isinstance(v, int):
                        item_dict[k] = GLib.Variant('i', v)
                remaining_packed.append(item_dict)

            if found:
                new_variant = GLib.Variant('aa{sv}', remaining_packed)
                settings.set_value("alarms", new_variant)
                _refresh_gnome_clocks_daemon()
                return f"Alarm matching '{name_or_id}' removed from GNOME Clocks."
        except Exception:
            pass

    # CLI Fallback
    alarms = _cli_get_alarms()
    filtered = []
    found = False
    for a in alarms:
        a_name = str(a.get("name", "")).lower()
        a_id = str(a.get("id", a.get("alarm-id", ""))).lower()
        if target == a_name or target == a_id or target in a_name:
            found = True
            continue
        filtered.append(a)

    if not found:
        return f"No alarm matching '{name_or_id}' was found."

    if _cli_set_alarms(filtered):
        _refresh_gnome_clocks_daemon()
        return f"Alarm matching '{name_or_id}' removed from GNOME Clocks."
    else:
        return f"Error removing alarm '{name_or_id}'."


def set_gnome_timer(duration_seconds: int, name: str = "Timer") -> str:
    """
    Sets a countdown timer in GNOME Clocks.
    """
    timer_name = name.strip() if name else "Timer"
    timer_id = f"orion_timer_{uuid.uuid4().hex[:8]}"

    if GSETTINGS_GIO_AVAILABLE:
        try:
            settings = Gio.Settings(schema_id="org.gnome.clocks")
            existing_variant = settings.get_value("timers")
            existing_timers: List[Dict[str, Any]] = existing_variant.unpack() if existing_variant else []

            new_timer = {
                'name': GLib.Variant('s', timer_name),
                'duration': GLib.Variant('d', float(duration_seconds)),
                'timer-id': GLib.Variant('s', timer_id),
                'state': GLib.Variant('i', 1),
            }

            packed_timers = []
            for item in existing_timers:
                item_dict = {}
                for k, v in item.items():
                    if isinstance(v, str):
                        item_dict[k] = GLib.Variant('s', v)
                    elif isinstance(v, (int, float)):
                        item_dict[k] = GLib.Variant('d', float(v))
                    elif isinstance(v, bool):
                        item_dict[k] = GLib.Variant('b', v)
                packed_timers.append(item_dict)

            packed_timers.append(new_timer)

            new_variant = GLib.Variant('aa{sv}', packed_timers)
            settings.set_value("timers", new_variant)
            _refresh_gnome_clocks_daemon()
            mins = duration_seconds // 60
            secs = duration_seconds % 60
            time_display = f"{mins}m {secs}s" if mins else f"{secs}s"
            return f"Timer '{timer_name}' for {time_display} started in GNOME Clocks."
        except Exception:
            pass

    # CLI Fallback
    timers = _cli_get_timers()
    timers.append({
        "name": timer_name,
        "duration": float(duration_seconds),
        "timer-id": timer_id,
        "state": 1
    })

    if _cli_set_timers(timers):
        _refresh_gnome_clocks_daemon()
        mins = duration_seconds // 60
        secs = duration_seconds % 60
        time_display = f"{mins}m {secs}s" if mins else f"{secs}s"
        return f"Timer '{timer_name}' for {time_display} started in GNOME Clocks."
    else:
        return "Error setting GNOME Clocks timer."
