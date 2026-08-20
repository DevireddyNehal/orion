import re
import queue
import time
import threading
import numpy as np
import sounddevice as sd
from time import perf_counter
from loguru import logger
from voice.providers import BaseTTSProvider, get_tts_provider

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002600-\U000026FF"  # Miscellaneous Symbols
    "]+",
    flags=re.UNICODE,
)


ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "eg", "ie",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "st", "ave", "rd", "blvd", "dept", "approx", "no", "vol"
}

# Major coordinating conjunctions and clause transitions for natural sub-sentence splitting
CLAUSE_CONJUNCTIONS = (
    "and", "but", "or", "so", "yet", "because", "since", "although",
    "while", "however", "though", "which", "where", "when", "as"
)


def clean_text_for_tts(text: str) -> str:
    """
    Sanitizes LLM text into clean, fluent, human-like spoken English for Neural TTS.
    Eliminates strange punctuation pauses, robotic glitches, and markdown artifacts.
    """
    if not text:
        return ""

    # 1. Strip emojis and pictographs
    text = EMOJI_PATTERN.sub("", text)
    # 2. Convert markdown links [title](url) to just title
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # 3. Strip bold, italics, code blocks, headers: *, _, #, `, ~, >
    text = re.sub(r'[*_#`~>]+', '', text)
    # 4. Convert em-dashes, en-dashes, and double hyphens into natural short comma pauses
    text = re.sub(r'[—–]|--', ', ', text)
    # 5. Remove quotes, parentheses, brackets, and math dollar signs (preserving intra-word apostrophes for contractions)
    text = re.sub(r'[\(\)\[\]\{\}"\$]', '', text)
    text = re.sub(r"(?<!\w)'|'(?!\w)", '', text)
    # 6. Normalize colons / semicolons that aren't inside numbers or timestamps (e.g. 10:30)
    text = re.sub(r'(?<!\d)[:;](?!\d)', ', ', text)
    # 7. Normalize spaces and newlines into single spaces
    text = re.sub(r'\s+', ' ', text)
    # 8. Clean up double punctuation (e.g. ", ," or ", .")
    text = re.sub(r',\s*,+', ', ', text)
    text = re.sub(r',\s*\.', '.', text)

    return text.strip()


def find_sentence_split_point(buffer: str) -> int:
    """
    Finds complete sentence boundaries (. ! ? or newline) with protection for abbreviations,
    numbers, and timestamps. Only splits long compound sentences if they exceed 18 words.
    """
    stripped = buffer.strip()
    if not stripped:
        return -1

    words = stripped.split()
    word_count = len(words)

    # 1. Search for sentence boundaries (. ! ? or newline)
    sentence_matches = list(re.finditer(r'([\.!\?\n])(?:\s+|$)', buffer))
    for m in sentence_matches:
        punct_idx = m.start(1)
        punct_char = buffer[punct_idx]

        # Guard against false sentence endings
        if punct_char == '.':
            # Check for decimals (e.g. 3.14, 72.5)
            if punct_idx > 0 and punct_idx + 1 < len(buffer):
                if buffer[punct_idx - 1].isdigit() and buffer[punct_idx + 1].isdigit():
                    continue

            # Check for abbreviations (e.g. Dr. Watson, Jan. 15, etc.)
            preceding_text = buffer[:punct_idx].rstrip()
            last_token = preceding_text.split()[-1] if preceding_text.split() else ""
            cleaned_token = re.sub(r'^[^\w]+|[^\w]+$', '', last_token).lower()
            if cleaned_token in ABBREVIATIONS:
                continue
            # Single capital letter (e.g. middle initial "A.")
            if len(last_token) == 1 and last_token.isupper():
                continue

        return m.end()

    # 2. Runaway Sentence Guard: If an unusually long sentence (>= 18 words or >= 90 chars) has no punctuation,
    # split at a natural conjunction boundary.
    if word_count >= 18 or len(buffer) >= 90:
        conjunction_pattern = r'(,\s+(?:' + '|'.join(CLAUSE_CONJUNCTIONS) + r')\b)'
        conj_matches = list(re.finditer(conjunction_pattern, buffer, flags=re.IGNORECASE))
        if conj_matches:
            for cm in conj_matches:
                pre_seg = buffer[:cm.start()].strip()
                if len(pre_seg.split()) >= 8:
                    return cm.start() + 1

    return -1


class TextToSpeech:
    """
    Decoupled general TTS Audio Playback & Streaming Engine.
    Handles text cleaning, full sentence prosody, and background playback.
    """
    def __init__(self, provider: BaseTTSProvider | None = None, sample_rate: int = 24000):
        t0 = perf_counter()
        if provider is None:
            self.provider = get_tts_provider()
        else:
            self.provider = provider

        logger.info(f"[TTS Engine] Initialized using provider: {self.provider.__class__.__name__} in {perf_counter() - t0:.2f}s")

        self.sample_rate = getattr(self.provider, "sample_rate", sample_rate)
        # Thread-safe queue to pass audio chunks to the sounddevice output stream
        self.audio_queue = queue.Queue()
        self.remainder = None
        self.starvation_start = None

        # Queue for phrases waiting to be synthesized by worker
        self.tts_queue = queue.Queue()

        # Dedicated background worker for provider synthesis
        self.worker_thread = threading.Thread(target=self._synthesis_worker, daemon=True)
        self.worker_thread.start()

        # Non-blocking background audio output stream
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            dtype='float32'
        )
        self.stream.start()

    def warmup(self):
        """Asynchronously pre-warms the TTS provider during the STT/LLM wait window."""
        if hasattr(self.provider, "warmup"):
            threading.Thread(target=self.provider.warmup, daemon=True).start()

    def get_buffered_duration(self) -> float:
        """Returns approximate seconds of unplayed audio currently buffered in the output queue."""
        samples = 0
        if self.remainder is not None:
            samples += len(self.remainder)
        with self.audio_queue.mutex:
            for item in self.audio_queue.queue:
                if item is not None and hasattr(item, "__len__"):
                    samples += len(item)
        return samples / float(self.sample_rate)

    def _synthesis_worker(self):
        """Single worker thread that consumes text phrases and synthesizes audio via provider."""
        while True:
            try:
                text = self.tts_queue.get()
                if text is None:
                    break
                logger.info(f"WORKER <<< {text}")
                self._synthesize_to_queue(text)
                self.tts_queue.task_done()
            except Exception as e:
                logger.error(f"[Worker Error] {e}")

    def _audio_callback(self, outdata, _, __, status):
        """Continuously feeds the sound card from the audio queue."""
        if status:
            logger.info(f"[Audio Status Warning] {status}")

        needed = len(outdata)
        out_buf = np.zeros(needed, dtype=np.float32)
        filled = 0

        # 1. Use leftover audio from previous callback
        if self.remainder is not None and len(self.remainder) > 0:
            take = min(needed - filled, len(self.remainder))
            out_buf[filled:filled + take] = self.remainder[:take]
            self.remainder = self.remainder[take:]
            filled += take

        # 2. Get audio chunks from queue
        while filled < needed:
            try:
                data = self.audio_queue.get_nowait()
            except queue.Empty:
                break

            if self.starvation_start is not None:
                duration = time.time() - self.starvation_start
                logger.info(f"[AUDIO] Audio resumed. Starved for {duration:.3f} seconds")
                self.starvation_start = None

            take = min(needed - filled, len(data))
            out_buf[filled:filled + take] = data[:take]
            filled += take

            if take < len(data):
                self.remainder = data[take:]
                break

        if filled == 0:
            outdata.fill(0)
            if self.starvation_start is None:
                self.starvation_start = time.time()
                logger.warning("[AUDIO] Queue starved")
        else:
            if filled < needed:
                out_buf[filled:] = 0
            outdata[:] = out_buf.reshape(-1, 1)

    def speak_stream(self, llm_generator) -> None:
        """
        Consumes tokens from the LLM generator.
        Streams complete natural sentences to the synthesizer for rich human-like prosody.
        """
        raw_buffer = ""

        for text_chunk, _ in llm_generator:
            logger.info(f"Received: {repr(text_chunk)}")
            raw_buffer += text_chunk

            while True:
                split_idx = find_sentence_split_point(raw_buffer)

                if split_idx == -1:
                    break

                sentence_raw = raw_buffer[:split_idx]
                raw_buffer = raw_buffer[split_idx:]

                clean_sentence = clean_text_for_tts(sentence_raw)
                if clean_sentence and any(c.isalnum() for c in clean_sentence):
                    logger.info(f"QUEUE >>> {clean_sentence}")
                    self.tts_queue.put(clean_sentence)

        # Handle any leftover trailing text at end of generation
        if raw_buffer.strip():
            clean_sentence = clean_text_for_tts(raw_buffer)
            if clean_sentence and any(c.isalnum() for c in clean_sentence):
                logger.info(f"QUEUE (FINAL) >>> {clean_sentence}")
                self.tts_queue.put(clean_sentence)

    def _synthesize_to_queue(self, text: str):
        """Delegates synthesis to provider and pushes float32 PCM audio chunks to playback queue."""
        logger.info(f"[SYNTH] START: {repr(text)}")
        try:
            t0 = time.time()
            chunk_count = 0
            first_audio = True

            for audio_data in self.provider.synthesize(text):
                chunk_count += 1
                if first_audio:
                    logger.info(f"[SYNTH] FIRST AUDIO CHUNK after {time.time() - t0:.3f}s")
                    first_audio = False

                if audio_data is not None and len(audio_data) > 0:
                    logger.info(f"[SYNTH] Queueing chunk #{chunk_count} ({len(audio_data)} samples)")
                    self.audio_queue.put(audio_data)

            logger.info(f"[SYNTH] DONE after {time.time() - t0:.3f}s ({chunk_count} chunks)")

        except Exception as e:
            logger.exception(f"[SYNTH ERROR] {e}")

    def close(self):
        """Stops the audio output stream cleanly when shutting down."""
        self.stream.stop()
        self.stream.close()