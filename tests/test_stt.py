import io
import unittest
import wave
from unittest.mock import MagicMock, patch

import numpy as np

from voice.stt import GroqSTT, LocalWhisperSTT, WhisperSTT


class TestSTT(unittest.TestCase):
    def test_groq_stt_missing_api_key_raises(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}, clear=True):
            with self.assertRaises(ValueError):
                GroqSTT()

    def test_groq_stt_transcribe_success(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "dummy_key"}, clear=True):
            with patch("groq.Groq") as mock_groq_cls:
                mock_client = MagicMock()
                mock_groq_cls.return_value = mock_client

                # Mock response
                mock_resp = MagicMock()
                mock_resp.text = "Good morning Orion"
                mock_client.audio.transcriptions.create.return_value = mock_resp

                stt = GroqSTT(api_key="dummy_key")

                # Generate dummy 16-bit PCM (0.1s of silence at 16000Hz)
                sample_rate = 16000
                dummy_samples = np.zeros(int(sample_rate * 0.1), dtype=np.int16)
                pcm_bytes = dummy_samples.tobytes()

                text = stt.transcribe(pcm_bytes, sample_rate=sample_rate)

                self.assertEqual(text, "Good morning Orion")
                mock_client.audio.transcriptions.create.assert_called_once()
                call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
                self.assertEqual(call_kwargs["language"], "en")
                self.assertEqual(call_kwargs["response_format"], "text")

                # Validate WAV file created in memory
                filename, wav_data = call_kwargs["file"]
                self.assertEqual(filename, "audio.wav")
                with wave.open(io.BytesIO(wav_data), "rb") as wf:
                    self.assertEqual(wf.getnchannels(), 1)
                    self.assertEqual(wf.getsampwidth(), 2)
                    self.assertEqual(wf.getframerate(), 16000)

    def test_local_whisper_stt_transcribe(self):
        with patch("faster_whisper.WhisperModel") as mock_model_cls:
            mock_model = MagicMock()
            mock_model_cls.return_value = mock_model

            # Mock segment returned by transcribe
            mock_seg1 = MagicMock()
            mock_seg1.text = "Hello there"
            mock_seg2 = MagicMock()
            mock_seg2.text = " how are you"
            mock_model.transcribe.return_value = ([mock_seg1, mock_seg2], None)

            local_stt = LocalWhisperSTT()

            # Create test int16 PCM
            raw_audio = np.array([0, 16384, -16384], dtype=np.int16).tobytes()
            result = local_stt.transcribe(raw_audio)

            self.assertEqual(result, "Hello there how are you")
            mock_model.transcribe.assert_called_once()
            called_audio = mock_model.transcribe.call_args[0][0]
            # Check float32 normalization
            self.assertTrue(np.allclose(called_audio, [0.0, 0.5, -0.5]))

    def test_whisper_stt_fallback_to_local_when_groq_fails(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "mock_groq_key", "STT_PROVIDER": "groq"}, clear=True):
            with patch("voice.stt.GroqSTT") as mock_groq_stt_cls:
                mock_groq = MagicMock()
                mock_groq_stt_cls.return_value = mock_groq
                mock_groq.transcribe.side_effect = RuntimeError("Groq Cloud Timeout")

                with patch("voice.stt.LocalWhisperSTT") as mock_local_cls:
                    mock_local = MagicMock()
                    mock_local_cls.return_value = mock_local
                    mock_local.transcribe.return_value = "Fallback transcription from local"

                    stt = WhisperSTT()
                    result = stt.transcribe(b"dummy_pcm")

                    mock_groq.transcribe.assert_called_once_with(b"dummy_pcm")
                    mock_local.transcribe.assert_called_once_with(b"dummy_pcm")
                    self.assertEqual(result, "Fallback transcription from local")

    def test_whisper_stt_uses_local_when_provider_is_local(self):
        with patch.dict("os.environ", {"STT_PROVIDER": "local", "GROQ_API_KEY": "has_key"}, clear=True):
            with patch("voice.stt.LocalWhisperSTT") as mock_local_cls:
                mock_local = MagicMock()
                mock_local_cls.return_value = mock_local
                mock_local.transcribe.return_value = "Local only speech"

                stt = WhisperSTT()
                self.assertIsNone(stt.groq_stt)
                result = stt.transcribe(b"dummy_pcm")
                self.assertEqual(result, "Local only speech")


if __name__ == "__main__":
    unittest.main()
