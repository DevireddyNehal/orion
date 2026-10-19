import time
import unittest
from unittest.mock import MagicMock

from tools.reminders import (
    ReminderManager,
    format_duration,
    parse_reminder_delay,
)


class TestReminders(unittest.TestCase):
    def test_parse_reminder_delay_relative(self):
        # Seconds
        self.assertEqual(parse_reminder_delay("in 30 seconds"), 30.0)
        self.assertEqual(parse_reminder_delay("10 secs"), 10.0)
        self.assertEqual(parse_reminder_delay("5s"), 5.0)

        # Minutes
        self.assertEqual(parse_reminder_delay("in 5 minutes"), 300.0)
        self.assertEqual(parse_reminder_delay("2 mins"), 120.0)
        self.assertEqual(parse_reminder_delay("1 minute"), 60.0)

        # Hours
        self.assertEqual(parse_reminder_delay("in 1 hour"), 3600.0)
        self.assertEqual(parse_reminder_delay("2 hrs"), 7200.0)

        # Word numbers
        self.assertEqual(parse_reminder_delay("in five minutes"), 300.0)
        self.assertEqual(parse_reminder_delay("in ten seconds"), 10.0)

        # Direct delay_seconds
        self.assertEqual(parse_reminder_delay(delay_seconds=45), 45.0)

    def test_parse_reminder_delay_clock_time(self):
        # Test clock time parsing returns positive seconds
        delay = parse_reminder_delay("11:59 PM")
        self.assertGreater(delay, 0)

    def test_format_duration(self):
        self.assertEqual(format_duration(30), "30 seconds")
        self.assertEqual(format_duration(300), "5 minutes")
        self.assertEqual(format_duration(3660), "1 hour 1 minute")

    def test_reminder_lifecycle_and_callback(self):
        manager = ReminderManager()
        mock_tts = MagicMock()
        manager.set_tts(mock_tts)

        # Set a short 0.2 second reminder
        res = manager.set_reminder(message="Check oven test", delay_seconds=0.2)
        self.assertIn("Reminder set: 'Check oven test'", res)

        # Verify listed in active reminders
        listed = manager.list_reminders()
        self.assertIn("Check oven test", listed)

        # Wait for timer to trigger (0.4 seconds)
        time.sleep(0.4)

        # Verify TTS was called with reminder message
        mock_tts.speak_text.assert_called_once_with(
            "Sir, here is your reminder: Check oven test."
        )

        # Verify reminder is no longer active
        self.assertIn("No active internal reminders", manager.list_reminders())

    def test_cancel_reminder(self):
        manager = ReminderManager()

        # Set reminder for 100 seconds
        res = manager.set_reminder(message="Call John test", delay_seconds=100)
        self.assertIn("Reminder set: 'Call John test'", res)

        # Cancel by keyword
        cancel_res = manager.cancel_reminder("John")
        self.assertIn("has been cancelled", cancel_res)

        # Verify no active reminders
        self.assertIn("No active internal reminders", manager.list_reminders())


if __name__ == "__main__":
    unittest.main()
