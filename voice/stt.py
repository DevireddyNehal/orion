import numpy as np
from faster_whisper import WhisperModel

class WhisperSTT:
    def __init__(self):
        model_size = "small"
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            local_files_only=True
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
