from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from tools.google_calendar import (
    create_calendar_event,
    delete_calendar_event,
    format_events_for_speech,
    get_local_timezone,
    get_today_agenda,
    list_upcoming_events,
    parse_calendar_datetime,
)
from tools.tool_registry import execute_tool


class TestGoogleCalendarUtilities(unittest.TestCase):
    def test_parse_calendar_datetime_relative_hours(self):
        tz = get_local_timezone()
        before = datetime.now(tz)
        start_dt, end_dt = parse_calendar_datetime("in 2 hours")
        after = datetime.now(tz)

        self.assertTrue(before + timedelta(hours=2) <= start_dt <= after + timedelta(hours=2))
        self.assertEqual(end_dt - start_dt, timedelta(hours=1))

    def test_parse_calendar_datetime_relative_minutes(self):
        tz = get_local_timezone()
        before = datetime.now(tz)
        start_dt, end_dt = parse_calendar_datetime("in 30 minutes")
        after = datetime.now(tz)

        self.assertTrue(before + timedelta(minutes=30) <= start_dt <= after + timedelta(minutes=30))
        self.assertEqual(end_dt - start_dt, timedelta(hours=1))

    def test_parse_calendar_datetime_tomorrow_meridiem(self):
        tz = get_local_timezone()
        now = datetime.now(tz)
        start_dt, end_dt = parse_calendar_datetime("tomorrow at 3pm")

        expected_date = (now + timedelta(days=1)).date()
        self.assertEqual(start_dt.date(), expected_date)
        self.assertEqual(start_dt.hour, 15)
        self.assertEqual(start_dt.minute, 0)
        self.assertEqual(end_dt - start_dt, timedelta(hours=1))

    def test_parse_calendar_datetime_noon(self):
        start_dt, end_dt = parse_calendar_datetime("today at noon")
        self.assertEqual(start_dt.hour, 12)
        self.assertEqual(start_dt.minute, 0)
        self.assertEqual(end_dt - start_dt, timedelta(hours=1))

    def test_format_events_for_speech_empty(self):
        msg = format_events_for_speech([], "today")
        self.assertEqual(msg, "Your calendar is completely clear today, sir.")

    def test_format_events_for_speech_single(self):
        events = [
            {"summary": "Sprint Planning", "formatted_time": "10:00 AM"}
        ]
        msg = format_events_for_speech(events, "today")
        self.assertIn("Sprint Planning", msg)
        self.assertIn("10:00 AM", msg)
        self.assertIn("one event", msg)

    def test_format_events_for_speech_multiple(self):
        events = [
            {"summary": "Sprint Planning", "formatted_time": "10:00 AM"},
            {"summary": "Design Sync", "formatted_time": "2:30 PM"},
        ]
        msg = format_events_for_speech(events, "today")
        self.assertIn("2 events today", msg)
        self.assertIn("Sprint Planning at 10:00 AM", msg)
        self.assertIn("Design Sync at 2:30 PM", msg)


class TestGoogleCalendarAPIOperations(unittest.TestCase):
    @patch("tools.google_calendar.get_calendar_service")
    def test_get_today_agenda_success(self, mock_get_service):
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        mock_events = {
            "items": [
                {
                    "id": "ev1",
                    "summary": "Team Sync",
                    "start": {"dateTime": "2026-10-20T10:00:00+05:30"},
                    "end": {"dateTime": "2026-10-20T10:30:00+05:30"},
                }
            ]
        }
        mock_service.events().list().execute.return_value = mock_events

        agenda = get_today_agenda()
        self.assertEqual(len(agenda), 1)
        self.assertEqual(agenda[0]["id"], "ev1")
        self.assertEqual(agenda[0]["summary"], "Team Sync")

    @patch("tools.google_calendar.get_calendar_service")
    def test_list_upcoming_events_success(self, mock_get_service):
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        mock_events = {
            "items": [
                {
                    "id": "ev2",
                    "summary": "Project Review",
                    "start": {"dateTime": "2026-10-21T14:00:00+05:30"},
                    "end": {"dateTime": "2026-10-21T15:00:00+05:30"},
                }
            ]
        }
        mock_service.events().list().execute.return_value = mock_events

        upcoming = list_upcoming_events(days=7)
        self.assertEqual(len(upcoming), 1)
        self.assertEqual(upcoming[0]["summary"], "Project Review")

    @patch("tools.google_calendar.get_calendar_service")
    def test_list_upcoming_events_with_query(self, mock_get_service):
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service
        mock_events = {
            "items": [
                {
                    "id": "ev3",
                    "summary": "Hackathon Demo",
                    "start": {"dateTime": "2026-10-25T10:00:00+05:30"},
                    "end": {"dateTime": "2026-10-25T11:00:00+05:30"},
                }
            ]
        }
        mock_service.events().list().execute.return_value = mock_events

        upcoming = list_upcoming_events(days=30, query="hackathon")
        self.assertEqual(len(upcoming), 1)
        self.assertEqual(upcoming[0]["summary"], "Hackathon Demo")

    @patch("tools.google_calendar.get_calendar_service")
    def test_create_calendar_event_success(self, mock_get_service):
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        mock_created = {
            "id": "new_ev_123",
            "summary": "AI Team Dinner",
            "htmlLink": "https://calendar.google.com/event?id=123",
        }
        mock_service.events().insert().execute.return_value = mock_created

        res = create_calendar_event(
            summary="AI Team Dinner",
            start_time="tomorrow at 7pm",
            description="Celebration dinner",
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["id"], "new_ev_123")
        self.assertEqual(res["summary"], "AI Team Dinner")

    @patch("tools.google_calendar.get_calendar_service")
    def test_delete_calendar_event_by_id(self, mock_get_service):
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service
        mock_service.events().delete().execute.return_value = {}

        res = delete_calendar_event("ev_abc123456789")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["deleted_id"], "ev_abc123456789")

    def test_execute_tool_dispatch_calendar(self):
        with patch("tools.tool_registry.TOOLS") as mock_tools:
            mock_func = MagicMock(return_value={"status": "success"})
            mock_tools.__contains__.side_effect = lambda k: k == "get_today_agenda"
            mock_tools.__getitem__.return_value = mock_func

            output = execute_tool("get_today_agenda", {})
            self.assertIn("success", output)


if __name__ == "__main__":
    unittest.main()
