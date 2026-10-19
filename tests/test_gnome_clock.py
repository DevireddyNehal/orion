import unittest
from datetime import datetime, timedelta

from tools.gnome_clock import parse_time


class TestGnomeClock(unittest.TestCase):
    def test_parse_time_12hr_ampm(self):
        # Morning times
        self.assertEqual(parse_time("7:30 AM"), (7, 30))
        self.assertEqual(parse_time("7:30 am"), (7, 30))
        self.assertEqual(parse_time("8 AM"), (8, 0))
        self.assertEqual(parse_time("12:00 AM"), (0, 0))
        self.assertEqual(parse_time("12:30 AM"), (0, 30))

        # Afternoon / Evening times
        self.assertEqual(parse_time("7:30 PM"), (19, 30))
        self.assertEqual(parse_time("8 PM"), (20, 0))
        self.assertEqual(parse_time("12:00 PM"), (12, 0))
        self.assertEqual(parse_time("12:45 PM"), (12, 45))
        self.assertEqual(parse_time("11:59 PM"), (23, 59))

    def test_parse_time_24hr(self):
        self.assertEqual(parse_time("00:00"), (0, 0))
        self.assertEqual(parse_time("07:15"), (7, 15))
        self.assertEqual(parse_time("19:45"), (19, 45))
        self.assertEqual(parse_time("23:59"), (23, 59))

    def test_parse_time_relative_words_and_digits(self):
        now = datetime.now()

        # Digits: 'in 15 minutes'
        expected_min = now + timedelta(minutes=15)
        h, m = parse_time("in 15 minutes")
        self.assertEqual((h, m), (expected_min.hour, expected_min.minute))

        # Words: 'in ten minutes'
        expected_word = now + timedelta(minutes=10)
        h, m = parse_time("in ten minutes")
        self.assertEqual((h, m), (expected_word.hour, expected_word.minute))

        # Hours: 'in 2 hours'
        expected_hr = now + timedelta(hours=2)
        h, m = parse_time("in 2 hours")
        self.assertEqual((h, m), (expected_hr.hour, expected_hr.minute))

        # Short forms: 'in 1 min', 'in 1 hr'
        expected_short = now + timedelta(minutes=1)
        h, m = parse_time("in 1 min")
        self.assertEqual((h, m), (expected_short.hour, expected_short.minute))

    def test_parse_time_invalid_formats(self):
        # Invalid hour bounds
        with self.assertRaises(ValueError):
            parse_time("25:00")

        # Invalid minute bounds
        with self.assertRaises(ValueError):
            parse_time("12:65")

        # Completely unparseable text
        with self.assertRaises(ValueError):
            parse_time("tomorrow sometime")

        with self.assertRaises(ValueError):
            parse_time("")


if __name__ == "__main__":
    unittest.main()
