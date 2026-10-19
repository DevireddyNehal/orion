import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from voice.providers import (
    EdgeTTSProvider,
    KokoroTTSProvider,
    PiperTTSProvider,
    get_tts_provider,
)


class TestTTSProviders(unittest.TestCase):
    def test_get_tts_provider_edge(self):
        with patch("voice.providers.EdgeTTSProvider") as mock_edge:
            p = get_tts_provider("edge_tts")
            mock_edge.assert_called_once()
            self.assertEqual(p, mock_edge.return_value)

    def test_get_tts_provider_kokoro(self):
        with patch("voice.providers.KokoroTTSProvider") as mock_kokoro:
            p = get_tts_provider("kokoro")
            mock_kokoro.assert_called_once()
            self.assertEqual(p, mock_kokoro.return_value)

    def test_get_tts_provider_piper(self):
        with patch("voice.providers.PiperTTSProvider") as mock_piper:
            p = get_tts_provider("piper")
            mock_piper.assert_called_once()
            self.assertEqual(p, mock_piper.return_value)

    def test_get_tts_provider_unknown_fallback(self):
        with patch("voice.providers.EdgeTTSProvider") as mock_edge:
            p = get_tts_provider("nonexistent_tts_engine")
            mock_edge.assert_called_once()
            self.assertEqual(p, mock_edge.return_value)

    def test_kokoro_voice_blending(self):
        mock_pipeline = MagicMock()
        mock_kokoro = MagicMock()
        mock_kokoro.KPipeline.return_value = mock_pipeline
        mock_pipeline.load_voice.side_effect = lambda name: np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with patch.dict("sys.modules", {"kokoro": mock_kokoro}), patch.dict("os.environ", {"TTS_VOICE": ""}):
            kokoro = KokoroTTSProvider(voice="0.6*bm_lewis+0.4*am_adam", lang_code="b")
            self.assertEqual(mock_pipeline.load_voice.call_count, 2)
            np.testing.assert_allclose(kokoro.voice, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_edge_tts_decode_empty(self):
        mock_edge = MagicMock()
        with patch.dict("sys.modules", {"edge_tts": mock_edge}):
            provider = EdgeTTSProvider()
            empty = provider._decode_mp3(b"")
            self.assertEqual(len(empty), 0)

    def test_kokoro_preset_resolution(self):
        mock_pipeline = MagicMock()
        mock_kokoro = MagicMock()
        mock_kokoro.KPipeline.return_value = mock_pipeline
        mock_pipeline.load_voice.side_effect = lambda name: np.array([1.0, 1.0], dtype=np.float32)

        with patch.dict("sys.modules", {"kokoro": mock_kokoro}), patch.dict("os.environ", {"TTS_VOICE": ""}):
            _ = KokoroTTSProvider(voice="orion", lang_code="b")
            calls = [c[0][0] for c in mock_pipeline.load_voice.call_args_list]
            self.assertIn("bm_lewis", calls)
            self.assertIn("bm_daniel", calls)

    def test_kokoro_trim_silence_padding(self):
        mock_pipeline = MagicMock()
        mock_kokoro = MagicMock()
        mock_kokoro.KPipeline.return_value = mock_pipeline
        mock_pipeline.load_voice.return_value = np.zeros(10)

        with patch.dict("sys.modules", {"kokoro": mock_kokoro}), patch.dict("os.environ", {"TTS_VOICE": ""}):
            kokoro = KokoroTTSProvider(voice="orion_classic")

            # 1. Empty array
            empty = np.array([], dtype=np.float32)
            self.assertEqual(len(kokoro._trim_silence_padding(empty)), 0)

            # 2. Pure silence returns original array
            silence = np.zeros(1000, dtype=np.float32)
            trimmed_silence = kokoro._trim_silence_padding(silence)
            self.assertEqual(len(trimmed_silence), 1000)

            # 3. Audio with 1000 samples silence prefix, 500 samples speech, 1000 samples silence suffix
            speech = np.ones(500, dtype=np.float32) * 0.5
            audio = np.concatenate([np.zeros(1000, dtype=np.float32), speech, np.zeros(1000, dtype=np.float32)])
            trimmed = kokoro._trim_silence_padding(audio, pad_samples=100)
            # Sliced range is [1000 - 100 : 1499 + 100] -> 1599 - 900 = 699 samples
            self.assertEqual(len(trimmed), 699)
            # Edge fade should fade start from 0 towards 0.5
            self.assertAlmostEqual(trimmed[0], 0.0, places=3)
            self.assertAlmostEqual(trimmed[-1], 0.0, places=3)

    def test_piper_missing_model_path_raises(self):
        mock_piper = MagicMock()
        with patch.dict("sys.modules", {"piper": mock_piper, "piper.config": mock_piper.config}):
            with patch.dict("os.environ", {"PIPER_MODEL_PATH": "", "TTS_VOICE": ""}, clear=True):
                with self.assertRaises(ValueError):
                    PiperTTSProvider(model_path="")

    def test_piper_nonexistent_model_path_raises(self):
        mock_piper = MagicMock()
        with patch.dict("sys.modules", {"piper": mock_piper, "piper.config": mock_piper.config}):
            with patch.dict("os.environ", {"PIPER_MODEL_PATH": "/path/to/missing/model.onnx"}, clear=True):
                with self.assertRaises(FileNotFoundError):
                    PiperTTSProvider()


if __name__ == "__main__":
    unittest.main()
