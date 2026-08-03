import numpy as np
import os

from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

class WhisperSTT:
    def __init__(self):
        model_size = os.getenv("STT_MODEL_SIZE", "small")
        local_only = os.getenv("STT_LOCAL_FILES_ONLY", "false").lower() in ("true", "1")
        try:
            self.model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                local_files_only=local_only
            )
        except Exception:
            self.model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                local_files_only=False
            )
    def transcribe(self, pcm_bytes: bytes) -> str:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        pcm_divisor = 32768.0
        audio = audio.astype(np.float32) / pcm_divisor
        segments, _ = self.model.transcribe(
            audio, 
            language="en",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True
            )
        
        return " ".join(s.text.strip() for s in segments)
