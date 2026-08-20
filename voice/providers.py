from abc import ABC, abstractmethod
import asyncio
import io
import os
import queue
import threading
import av
import numpy as np
from loguru import logger
from typing import Any, Generator

class BaseTTSProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> Generator[np.ndarray, None, None]:
        """Yields float32 numpy audio arrays at target sample rate for sounddevice."""
        pass

    def warmup(self):
        """Optional provider pre-warming."""
        pass


class KokoroTTSProvider(BaseTTSProvider):
    def __init__(self, voice: str = "orion", lang_code: str | None = None, speed: float | None = None):
        from kokoro import KPipeline

        self.raw_voice_name = os.getenv("TTS_VOICE", voice)
        lang = os.getenv("TTS_LANG", lang_code)
        self.speed = float(os.getenv("TTS_SPEED", str(speed if speed is not None else 1.2)))

        # Auto-detect language code if not explicitly set
        if not lang:
            v_lower = self.raw_voice_name.lower()
            if "bm_" in v_lower or "bf_" in v_lower or "orion" in v_lower:
                lang = "b"
            else:
                lang = "a"

        logger.info(f"[KokoroTTS] Initializing Kokoro Pipeline (lang='{lang}', speed={self.speed})...")
        self.pipeline = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device="cpu")
        self.sample_rate = 24000
        self.voice: Any = self._load_voice_or_blend(self.raw_voice_name)
        logger.info(f"[KokoroTTS] Initialized with voice: {self.raw_voice_name} (speed={self.speed})")

        # Warmup
        try:
            list(self.pipeline(" ", voice=self.voice, speed=self.speed))  # type: ignore
            logger.info("[KokoroTTS] Warm-up successful.")
        except Exception as e:
            logger.error(f"[KokoroTTS] Warm-up failed: {e}")

    def warmup(self):
        """Quickly keeps the pipeline and PyTorch kernels warm in CPU cache."""
        try:
            list(self.pipeline(" ", voice=self.voice, speed=self.speed))  # type: ignore
        except Exception:
            pass

    def _load_voice_or_blend(self, voice_str: str) -> Any:
        voice_str = voice_str.strip()
        presets = {
            # Modern, young, crisp AI assistant blends (energetic & natural)
            "orion": "0.65*bm_lewis+0.35*bm_daniel",
            "orion_crisp": "0.65*bm_lewis+0.35*bm_daniel",
            "orion_young_tech": "0.60*bm_lewis+0.40*am_adam",
            "orion_dynamic": "0.70*bm_daniel+0.30*bm_fable",
            "orion_lewis": "bm_lewis",
            "orion_classic": "0.65*bm_george+0.35*bm_lewis",
        }

        if voice_str.lower() in presets:
            logger.info(f"[KokoroTTS] Using voice preset '{voice_str}' -> {presets[voice_str.lower()]}")
            voice_str = presets[voice_str.lower()]

        if "+" in voice_str or "*" in voice_str:
            parts = [p.strip() for p in voice_str.split("+")]
            blended_tensor = None
            for part in parts:
                if "*" in part:
                    weight_str, name = [x.strip() for x in part.split("*", 1)]
                    weight = float(weight_str)
                else:
                    weight = 1.0 / len(parts)
                    name = part
                v_tensor = self.pipeline.load_voice(name)
                if blended_tensor is None:
                    blended_tensor = weight * v_tensor
                else:
                    blended_tensor = blended_tensor + weight * v_tensor
            return blended_tensor
        else:
            return self.pipeline.load_voice(voice_str)

    def _trim_silence_padding(self, audio: np.ndarray, threshold: float = 0.005, pad_samples: int = 960) -> np.ndarray:
        """Trims artificial dead silence padding from Kokoro chunks while keeping a ~40ms natural breath cushion."""
        if len(audio) == 0:
            return audio
        non_silent = np.where(np.abs(audio) > threshold)[0]
        if len(non_silent) == 0:
            return audio
        start = max(0, non_silent[0] - pad_samples)
        end = min(len(audio), non_silent[-1] + pad_samples)
        return audio[start:end]

    def synthesize(self, text: str) -> Generator[np.ndarray, None, None]:
        audio_generator = self.pipeline(text, voice=self.voice, speed=self.speed)  # type: ignore
        for _, _, audio in audio_generator:
            if audio is not None and len(audio) > 0:
                trimmed = self._trim_silence_padding(np.asarray(audio, dtype=np.float32))
                if len(trimmed) > 0:
                    yield trimmed


class EdgeTTSProvider(BaseTTSProvider):
    def __init__(self, voice: str = "en-US-EmmaNeural", target_sample_rate: int = 24000):
        import edge_tts
        self.edge_tts = edge_tts
        self.voice = os.getenv("TTS_VOICE", voice)
        self.sample_rate = target_sample_rate

        # Persistent background asyncio event loop
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

        logger.info(f"[EdgeTTS] Initialized with voice: {self.voice} and persistent async loop")

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _decode_mp3(self, mp3_bytes: bytes) -> np.ndarray:
        if not mp3_bytes:
            return np.array([], dtype=np.float32)
        try:
            container = av.open(io.BytesIO(mp3_bytes), mode='r')
            if not container.streams.audio:
                return np.array([], dtype=np.float32)
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(
                format='flt',
                layout='mono',
                rate=self.sample_rate
            )
            samples = []
            for frame in container.decode(stream):
                resampled_frames = resampler.resample(frame)
                if resampled_frames:
                    for r_frame in resampled_frames:
                        arr = r_frame.to_ndarray().flatten()
                        samples.append(arr)
            flushed = resampler.resample(None)
            if flushed:
                for r_frame in flushed:
                    arr = r_frame.to_ndarray().flatten()
                    samples.append(arr)
            container.close()
            if not samples:
                return np.array([], dtype=np.float32)
            return np.concatenate(samples)
        except Exception as e:
            logger.error(f"[EdgeTTS] MP3 decode error: {e}")
            return np.array([], dtype=np.float32)

    async def _async_synthesize(self, text: str, out_queue: queue.Queue):
        try:
            communicate = self.edge_tts.Communicate(text, self.voice)
            mp3_data = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_data.extend(chunk["data"])

            if mp3_data:
                pcm_data = self._decode_mp3(bytes(mp3_data))
                if len(pcm_data) > 0:
                    out_queue.put(pcm_data)
        except Exception as e:
            logger.error(f"[EdgeTTS] Synthesis error: {e}")
        finally:
            out_queue.put(None)

    def synthesize(self, text: str) -> Generator[np.ndarray, None, None]:
        if not text or not text.strip():
            return

        out_queue = queue.Queue()
        asyncio.run_coroutine_threadsafe(
            self._async_synthesize(text, out_queue),
            self.loop
        )

        while True:
            item = out_queue.get()
            if item is None:
                break
            yield item


class PiperTTSProvider(BaseTTSProvider):
    def __init__(self, model_path: str | None = None, config_path: str | None = None):
        from piper import PiperVoice
        from piper.config import SynthesisConfig

        self.model_path = model_path or os.getenv("PIPER_MODEL_PATH") or os.getenv("TTS_VOICE") or ""

        if not self.model_path:
            raise ValueError(
                "[PiperTTS] No model path specified. Set PIPER_MODEL_PATH in your .env "
                "or place an ONNX model in voice/models/."
            )

        if not os.path.exists(self.model_path):
            candidate = os.path.join("voice", "models", self.model_path)
            if os.path.exists(candidate):
                self.model_path = candidate
            elif os.path.exists(f"{candidate}.onnx"):
                self.model_path = f"{candidate}.onnx"
            else:
                raise FileNotFoundError(
                    f"[PiperTTS] Model file not found at '{self.model_path}'. "
                    "Ensure your custom ONNX voice model exists."
                )

        self.config_path = config_path or os.getenv("PIPER_CONFIG_PATH")
        if not self.config_path:
            json_candidate = f"{self.model_path}.json"
            if os.path.exists(json_candidate):
                self.config_path = json_candidate

        logger.info(f"[PiperTTS] Loading Piper model from: {self.model_path}...")
        self.voice = PiperVoice.load(self.model_path, config_path=self.config_path)
        self.sample_rate = self.voice.config.sample_rate

        # Tune synthesis parameters to eliminate robotic drawl (default model json uses 1.25 length_scale)
        length_scale = float(os.getenv("PIPER_LENGTH_SCALE", "1.0"))
        noise_scale = float(os.getenv("PIPER_NOISE_SCALE", "0.667"))
        noise_w_scale = float(os.getenv("PIPER_NOISE_W", "0.8"))

        self.syn_config = SynthesisConfig(
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
        )

        logger.info(
            f"[PiperTTS] Initialized Piper model successfully (sample_rate={self.sample_rate}, "
            f"length_scale={length_scale}, noise_scale={noise_scale}, noise_w_scale={noise_w_scale})"
        )

    def synthesize(self, text: str) -> Generator[np.ndarray, None, None]:
        if not text or not text.strip():
            return

        for chunk in self.voice.synthesize(text, syn_config=self.syn_config):
            if chunk.audio_float_array is not None and len(chunk.audio_float_array) > 0:
                yield chunk.audio_float_array


def get_tts_provider(provider_name: str | None = None) -> BaseTTSProvider:
    if not provider_name:
        provider_name = os.getenv("TTS_PROVIDER", "edge_tts")
    provider_name = provider_name.lower().strip()
    
    if provider_name in ("edge", "edge_tts", "edgetts"):
        return EdgeTTSProvider()
    elif provider_name in ("kokoro", "local"):
        return KokoroTTSProvider()
    elif provider_name in ("piper", "piper_tts", "pipertts"):
        return PiperTTSProvider()
    else:
        logger.warning(f"Unknown TTS_PROVIDER '{provider_name}'. Defaulting to EdgeTTSProvider.")
        return EdgeTTSProvider()

