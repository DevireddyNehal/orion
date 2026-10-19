import unittest
from unittest.mock import MagicMock, patch

from evdev import ecodes

from voice.recorder import AudioRecorder


class TestAudioRecorder(unittest.TestCase):
    @patch("voice.recorder.AudioRecorder._discover_keyboards", return_value=[])
    @patch("threading.Thread")
    def test_default_key_resolution(self, mock_thread, mock_discover):
        with patch.dict("os.environ", {}, clear=True):
            recorder = AudioRecorder()
            self.assertEqual(recorder.target_key_name, "KEY_RIGHTCTRL")
            self.assertEqual(recorder.target_key_code, ecodes.KEY_RIGHTCTRL)

    @patch("voice.recorder.AudioRecorder._discover_keyboards", return_value=[])
    @patch("threading.Thread")
    def test_custom_key_resolution(self, mock_thread, mock_discover):
        with patch.dict("os.environ", {"RECORD_KEY": "KEY_LEFTCTRL"}):
            recorder = AudioRecorder()
            self.assertEqual(recorder.target_key_name, "KEY_LEFTCTRL")
            self.assertEqual(recorder.target_key_code, ecodes.KEY_LEFTCTRL)

        with patch.dict("os.environ", {"RECORD_KEY": "KEY_SCROLLLOCK"}):
            recorder = AudioRecorder()
            self.assertEqual(recorder.target_key_name, "KEY_SCROLLLOCK")
            self.assertEqual(recorder.target_key_code, ecodes.KEY_SCROLLLOCK)

    @patch("voice.recorder.AudioRecorder._discover_keyboards", return_value=[])
    @patch("threading.Thread")
    def test_unknown_key_resolution_defaults(self, mock_thread, mock_discover):
        with patch.dict("os.environ", {"RECORD_KEY": "NONEXISTENT_KEY_XYZ"}):
            recorder = AudioRecorder()
            self.assertEqual(recorder.target_key_name, "KEY_RIGHTCTRL")
            self.assertEqual(recorder.target_key_code, ecodes.KEY_RIGHTCTRL)

    @patch("voice.recorder.AudioRecorder._discover_keyboards", return_value=[])
    @patch("threading.Thread")
    def test_register_on_press_callback(self, mock_thread, mock_discover):
        recorder = AudioRecorder()
        dummy_cb = MagicMock()
        recorder.register_on_press_callback(dummy_cb)

        self.assertIn(dummy_cb, recorder.on_press_callbacks)

        # Ensure registering same callback twice does not duplicate
        recorder.register_on_press_callback(dummy_cb)
        self.assertEqual(recorder.on_press_callbacks.count(dummy_cb), 1)


if __name__ == "__main__":
    unittest.main()
