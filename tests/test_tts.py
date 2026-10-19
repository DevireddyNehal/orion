import unittest

from voice.tts import (
    clean_text_for_tts,
    determine_sentence_pacing,
    find_sentence_split_point,
)


class TestTTSUtilities(unittest.TestCase):
    def test_clean_text_basic(self):
        self.assertEqual(clean_text_for_tts("Hello, world!"), "Hello, world!")
        self.assertEqual(clean_text_for_tts(""), "")
        self.assertEqual(clean_text_for_tts(None), "")

    def test_clean_text_emoji_removal(self):
        text = "Hello sir! 😊 The weather is great. 🚀☀️"
        cleaned = clean_text_for_tts(text)
        self.assertNotIn("😊", cleaned)
        self.assertNotIn("🚀", cleaned)
        self.assertNotIn("☀️", cleaned)
        self.assertEqual(cleaned, "Hello sir! The weather is great.")

    def test_clean_text_latex_math_removal(self):
        text = r"The equation is \(E = mc^2\) and \[F = ma\] with \pi and \alpha"
        cleaned = clean_text_for_tts(text)
        self.assertNotIn(r"\(", cleaned)
        self.assertNotIn(r"\[", cleaned)
        self.assertNotIn(r"\pi", cleaned)
        self.assertNotIn(r"\alpha", cleaned)
        self.assertIn("The equation is and with and", cleaned)

    def test_clean_text_markdown_formatting(self):
        text = "This is **bold**, this is *italic*, and `code` here."
        cleaned = clean_text_for_tts(text)
        self.assertEqual(cleaned, "This is bold, this is italic, and code here.")

    def test_clean_text_markdown_links(self):
        text = "Check [Google Search](https://google.com) for answers."
        cleaned = clean_text_for_tts(text)
        self.assertEqual(cleaned, "Check Google Search for answers.")

    def test_clean_text_list_bullets(self):
        text = "- First item\n* Second item\n- Third item"
        cleaned = clean_text_for_tts(text)
        self.assertNotIn("- ", cleaned)
        self.assertNotIn("* ", cleaned)
        self.assertEqual(cleaned, "First item Second item Third item")

    def test_clean_text_dashes_to_commas(self):
        text = "We have three options—or perhaps four--or two – none."
        cleaned = clean_text_for_tts(text)
        self.assertNotIn("—", cleaned)
        self.assertNotIn("–", cleaned)
        self.assertNotIn("--", cleaned)
        self.assertIn("three options, or perhaps", cleaned)

    def test_clean_text_colons_and_timestamps(self):
        # Timestamps should preserve colon (10:30)
        # Structural colons should convert to commas
        text = "At 10:30 AM, we begin: the project launch."
        cleaned = clean_text_for_tts(text)
        self.assertIn("10:30", cleaned)
        self.assertIn("begin, the project launch", cleaned)

    def test_determine_sentence_pacing(self):
        # Empty sentence
        self.assertEqual(determine_sentence_pacing(""), 1.0)

        # Short non-technical sentence (< 12 words) -> 1.20x
        short = "Alarm set for 7:30 AM, sir."
        self.assertEqual(determine_sentence_pacing(short), 1.20)

        # Medium non-technical sentence (< 22 words) -> 1.10x
        medium = "I have checked your schedule for today and everything appears to be in order."
        self.assertEqual(determine_sentence_pacing(medium), 1.10)

        # Technical/explanatory sentence with keywords -> 0.90x - 0.95x
        tech_short = "The quantum algorithm optimizes the state."
        self.assertLessEqual(determine_sentence_pacing(tech_short), 0.95)

        tech_long = "In general relativity the curvature of spacetime is directly related to the energy and momentum of whatever matter and radiation are present in the gravitational field."
        self.assertEqual(determine_sentence_pacing(tech_long), 0.90)

    def test_find_sentence_split_point_full_sentences(self):
        buffer = "Hello sir! How can I help you today?"
        idx = find_sentence_split_point(buffer)
        self.assertGreater(idx, 0)
        self.assertEqual(buffer[:idx].strip(), "Hello sir!")

    def test_find_sentence_split_point_preserves_decimals(self):
        buffer = "The value is 3.14159 approximately."
        idx = find_sentence_split_point(buffer)
        # Should not split on the period in 3.14159
        self.assertEqual(buffer[:idx], "The value is 3.14159 approximately.")

    def test_find_sentence_split_point_preserves_abbreviations(self):
        buffer = "Dr. Watson examined the sample."
        idx = find_sentence_split_point(buffer)
        # Should not split on Dr.
        self.assertEqual(buffer[:idx], "Dr. Watson examined the sample.")


if __name__ == "__main__":
    unittest.main()
