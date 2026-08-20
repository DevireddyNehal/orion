import io
import os
import wave
import numpy as np
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class LocalWhisperSTT:
    def __init__(self):
        from faster_whisper import WhisperModel
        model_size = os.getenv("STT_MODEL_SIZE", "base.en")
        local_only = os.getenv("STT_LOCAL_FILES_ONLY", "false").lower() in ("true", "1")
        logger.info(f"[LocalSTT] Initializing faster-whisper ('{model_size}')...")
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
        logger.info("[LocalSTT] Local Whisper initialized successfully.")

    def transcribe(self, pcm_bytes: bytes) -> str:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        pcm_divisor = 32768.0
        audio = audio.astype(np.float32) / pcm_divisor
        beam_size = int(os.getenv("STT_BEAM_SIZE", "2"))
        segments, _ = self.model.transcribe(
            audio, 
            language="en",
            beam_size=beam_size,
            vad_filter=True,
            condition_on_previous_text=False
        )
        return " ".join(s.text.strip() for s in segments)


class GroqSTT:
    def __init__(self, api_key: str | None = None):
        from groq import Groq
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("[GroqSTT] GROQ_API_KEY not found.")
        self.client = Groq(api_key=self.api_key)
        self.model = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
        logger.info(f"[GroqSTT] Initialized Groq STT with model '{self.model}'.")

    def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        buf.seek(0)

        response = self.client.audio.transcriptions.create(
            file=("audio.wav", buf.read()),
            model=self.model,
            language="en",
            response_format="text"
        )
        if isinstance(response, str):
            return response.strip()
        return getattr(response, "text", str(response)).strip()


class WhisperSTT:
    """
    Unified STT Interface.
    Routes to GroqSTT if configured / available, with automatic fallback to LocalWhisperSTT.
    """
    def __init__(self):
        stt_provider = os.getenv("STT_PROVIDER", "").lower().strip()
        groq_key = os.getenv("GROQ_API_KEY", "").strip()

        self.groq_stt: GroqSTT | None = None
        self.local_stt: LocalWhisperSTT | None = None

        if stt_provider == "local":
            logger.info("[STT] STT_PROVIDER is set to 'local'. Using on-device Whisper.")
            self.local_stt = LocalWhisperSTT()
        elif groq_key or stt_provider == "groq":
            try:
                self.groq_stt = GroqSTT(api_key=groq_key)
            except Exception as e:
                logger.warning(f"[STT] Could not initialize Groq STT ({e}). Falling back to local Whisper.")
                self.local_stt = LocalWhisperSTT()
        else:
            logger.info("[STT] No GROQ_API_KEY found. Defaulting to on-device Whisper.")
            self.local_stt = LocalWhisperSTT()

    def _ensure_local(self) -> LocalWhisperSTT:
        if self.local_stt is None:
            logger.info("[STT] Loading local fallback Whisper model...")
            self.local_stt = LocalWhisperSTT()
        return self.local_stt

    def transcribe(self, pcm_bytes: bytes) -> str:
        if self.groq_stt is not None:
            try:
                return self.groq_stt.transcribe(pcm_bytes)
            except Exception as e:
                logger.error(f"[GroqSTT Error] {e}. Falling back to local Whisper...")
                return self._ensure_local().transcribe(pcm_bytes)
        else:
            return self._ensure_local().transcribe(pcm_bytes)

