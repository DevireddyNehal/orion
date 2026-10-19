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
        self.assertEqual(buffer[:idx], "Dr. Watson examined the sample.")

    def test_find_sentence_split_point_preserves_first_sentence_without_mid_thought_splits(self):
        # 8-word sentence is preserved intact for first chunk
        s1 = "Torque is the twisting force causing angular rotation."
        idx = find_sentence_split_point(s1, is_first_chunk=True)
        self.assertEqual(s1[:idx].strip(), s1.strip())

        # For subsequent chunks, up to 16 words are preserved intact
        s2 = "Torque, sir, is the twisting force that makes an object rotate around an axis."
        idx2 = find_sentence_split_point(s2, is_first_chunk=False)
        self.assertEqual(s2[:idx2].strip(), s2.strip())

    def test_find_sentence_split_point_splits_long_run_on_sentences_at_clauses(self):
        s2 = (
            "Imagine pushing a door, the farther your hand is from the hinges, "
            "the easier it is to swing the door open, that distance multiplied by "
            "the strength of your push is the torque."
        )
        # First split should happen at the second comma ('hinges,')
        idx1 = find_sentence_split_point(s2, is_first_chunk=False)
        self.assertIn("hinges,", s2[:idx1])

        # Second split should happen at 'open,'
        rem2 = s2[idx1:].strip()
        idx2 = find_sentence_split_point(rem2, is_first_chunk=False)
        self.assertIn("open,", rem2[:idx2])

        # Third split should complete the remainder
        rem3 = rem2[idx2:].strip()
        idx3 = find_sentence_split_point(rem3, is_first_chunk=False)
        self.assertEqual(rem3[:idx3].strip(), rem3.strip())

    def test_find_sentence_split_point_merges_short_openers_and_caps_first_chunk(self):
        text = "Certainly, sir. Torque is the twisting force that makes an object rotate around an axis, much like how you turn a wrench."
        idx = find_sentence_split_point(text, is_first_chunk=True)
        # Merges 'Certainly, sir.' with introductory clause up to 'force', capping at 7 words (~1.5s TTFA)
        self.assertIn("Certainly, sir.", text[:idx])
        self.assertIn("twisting force", text[:idx])
        self.assertNotIn("axis,", text[:idx])

        # Chunk 2 takes the next balanced clause up to 'axis,'
        rem = text[idx:].strip()
        idx2 = find_sentence_split_point(rem, is_first_chunk=False)
        self.assertIn("axis,", rem[:idx2])

    def test_find_sentence_split_point_splits_long_clause_before_twenty_words(self):
        text = (
            "Special relativity tells us that the laws of physics are the same "
            "for all observers moving at constant speed, and that the speed of "
            "light is the ultimate speed limit."
        )
        idx = find_sentence_split_point(text, is_first_chunk=False)
        self.assertIn("constant speed,", text[:idx])


if __name__ == "__main__":
    unittest.main()
