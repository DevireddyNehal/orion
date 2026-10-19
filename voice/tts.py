import queue
import re
import threading
import time
from time import perf_counter

import numpy as np
import sounddevice as sd
from loguru import logger

from voice.providers import BaseTTSProvider, get_tts_provider

EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
    "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
    "\U00002600-\U000026ff"  # Miscellaneous Symbols
    "]+",
    flags=re.UNICODE,
)


ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "eg",
    "ie",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
    "st",
    "ave",
    "rd",
    "blvd",
    "dept",
    "approx",
    "no",
    "vol",
}

# Major coordinating conjunctions and clause transitions for natural sub-sentence splitting
CLAUSE_CONJUNCTIONS = (
    "and",
    "but",
    "or",
    "so",
    "yet",
    "because",
    "since",
    "although",
    "while",
    "however",
    "though",
    "which",
    "where",
    "when",
    "as",
    "that",
    "who",
)


def clean_text_for_tts(text: str | None) -> str:
    """
    Sanitizes LLM text into clean, fluent, human-like spoken English for Neural TTS.
    Eliminates strange punctuation pauses, robotic glitches, LaTeX math, and markdown artifacts.
    """
    if not text:
        return ""

    # 1. Strip emojis and pictographs
    text = EMOJI_PATTERN.sub("", text)

    # 2. Strip LaTeX math expressions (e.g. \(...\), \[...\], \G\mu\nu, \pi, etc.)
    text = re.sub(r"\\\([^)]*\\\)", "", text)
    text = re.sub(r"\\\[[^\]]*\\\]", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)

    # 3. Convert markdown links [title](url) to just title
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # 4. Strip leading list bullets, hyphens, numbers at start of line/sentence (e.g. "- ", "* ", "1. ", "• ")
    text = re.sub(r"^\s*[-*•]\s+", "", text)
    text = re.sub(r"^\s*\d+[\.\)]\s+", "", text)
    text = re.sub(r"(?<=\s)[-*•]\s+", "", text)

    # 5. Strip bold, italics, code blocks, headers: *, _, #, `, ~, >
    text = re.sub(r"[*_#`~>]+", "", text)

    # 6. Convert em-dashes, en-dashes, and double hyphens into natural short comma pauses
    text = re.sub(r"[—–]|--", ", ", text)

    # 7. Remove quotes, backslashes, parentheses, brackets, and math dollar signs
    text = re.sub(r'[\(\)\[\]\{\}"\$\\]', "", text)
    text = re.sub(r"(?<!\w)'|'(?!\w)", "", text)

    # 8. Normalize colons / semicolons that aren't inside numbers or timestamps (e.g. 10:30)
    text = re.sub(r"(?<!\d)[:;](?!\d)", ", ", text)

    # 9. Normalize spaces and newlines into single spaces
    text = re.sub(r"\s+", " ", text)

    # 10. Clean up leading comma/punctuation or double punctuation (e.g. ", ," or ", .")
    text = re.sub(r"^[\s,]+", "", text)
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r",\s*\.", ".", text)

    return text.strip()


def find_sentence_split_point(
    buffer: str, is_first_chunk: bool = False, buffered_duration: float = 0.0
) -> int:
    """
    Finds natural sentence and clause split points for streaming TTS.
    - Merges tiny conversational openers (<= 4 words like 'Certainly, sir.') with the following clause
      for the first chunk so the soundcard starts with a strong audio runway.
    - Preserves full prosody by keeping complete sentences intact when <= 16 words.
    - If a sentence is long (>= 17 words) with clause punctuation (commas, dashes, colons),
      splits at a balanced clause boundary (7-13 words) to ensure CPU synthesis speed stays
      ahead of audio playback, preventing queue starvation.
    - For initial chunks without full sentence endings, only splits at legitimate punctuation
      boundaries (>= 6 words); never splits mid-thought across arbitrary unpunctuated whitespace.
    """
    stripped = buffer.strip()
    if not stripped:
        return -1

    words = stripped.split()
    word_count = len(words)

    # 1. Check for complete sentence boundaries (. ! ? \n)
    sentence_matches = list(re.finditer(r"([\.!\?\n])(?:\s+|$)", buffer))
    valid_sentence_match = None

    for m in sentence_matches:
        punct_idx = m.start(1)
        punct_char = buffer[punct_idx]

        # Guard against false sentence endings: decimals (e.g. 3.14)
        if punct_char == ".":
            if (
                punct_idx > 0
                and punct_idx + 1 < len(buffer)
                and buffer[punct_idx - 1].isdigit()
                and buffer[punct_idx + 1].isdigit()
            ):
                continue

            # Check for abbreviations (e.g. Dr. Watson, Jan. 15, etc.)
            preceding_text = buffer[:punct_idx].rstrip()
            last_token = preceding_text.split()[-1] if preceding_text.split() else ""
            cleaned_token = re.sub(r"^[^\w]+|[^\w]+$", "", last_token).lower()
            if cleaned_token in ABBREVIATIONS:
                continue
            # Single capital letter (e.g. middle initial "A.")
            if len(last_token) == 1 and last_token.isupper():
                continue

        # For the first chunk, merge tiny conversational openers (<= 4 words, e.g. "Certainly, sir.")
        # so the audio queue begins with a substantial buffer rather than a 0.9s starvation trap.
        cand_words = len(buffer[: m.end()].split())
        if is_first_chunk and cand_words <= 4:
            continue

        valid_sentence_match = m
        break

    if valid_sentence_match is not None:
        candidate_sentence = buffer[: valid_sentence_match.end()]
        sentence_words = candidate_sentence.split()
        n_sentence_words = len(sentence_words)

        # For the first chunk, target 6-9 words for fast ~1.5s TTFA without starvation.
        # For subsequent chunks, allow up to 16 words for complete natural sentence prosody.
        max_words = 9 if is_first_chunk else 16
        if n_sentence_words <= max_words:
            return valid_sentence_match.end()

        # If sentence exceeds max_words, check for natural clause breaks (, ; : —)
        clause_matches = list(
            re.finditer(r"([,;:—–]|\s+--\s+)(?:\s+|$)", candidate_sentence)
        )
        for cm in clause_matches:
            pre_seg = candidate_sentence[: cm.start(1)].strip()
            pre_count = len(pre_seg.split())
            rem_count = n_sentence_words - pre_count
            target_min = 6 if is_first_chunk else 7
            target_max = 9 if is_first_chunk else 13
            if target_min <= pre_count <= target_max and rem_count >= 4:
                return cm.end()

        # If it's the first chunk and no clause punctuation fell in 6-9 words,
        # split at a natural conjunction (that, which, because, since, and, but, so)
        if is_first_chunk:
            conj_pattern = r"(\s+(?:" + "|".join(CLAUSE_CONJUNCTIONS) + r")\b)"
            for conj in re.finditer(conj_pattern, candidate_sentence, flags=re.IGNORECASE):
                pre_seg = candidate_sentence[: conj.start()].strip()
                pre_count = len(pre_seg.split())
                if 6 <= pre_count <= 9:
                    return conj.start() + 1

        # Fallback clause search:
        for cm in clause_matches:
            pre_seg = candidate_sentence[: cm.start(1)].strip()
            pre_count = len(pre_seg.split())
            target_limit = 10 if is_first_chunk else 20
            if 6 <= pre_count <= target_limit:
                return cm.end()

        # Fallback to full sentence if no suitable clause punctuation was found
        return valid_sentence_match.end()

    # 2. No complete sentence punctuation in buffer yet (LLM still streaming current sentence):
    # Check if we can split an early clause (comma, dash, colon)
    clause_match = re.search(r"([,;:—–]|\s+--\s+)(?:\s+|$)", buffer)
    if clause_match:
        pre_text = buffer[: clause_match.start(1)].strip()
        pre_words = len(pre_text.split())
        post_words = len(buffer[clause_match.end() :].split())
        target_min = 6 if is_first_chunk else 7
        target_max = 9 if is_first_chunk else 14
        if target_min <= pre_words <= target_max and post_words >= 1:
            return clause_match.end()

    # For first chunk without punctuation yet: check conjunction in 6-9 words
    if is_first_chunk and word_count >= 7:
        conj_pattern = r"(\s+(?:" + "|".join(CLAUSE_CONJUNCTIONS) + r")\b)"
        for conj in re.finditer(conj_pattern, buffer, flags=re.IGNORECASE):
            pre_seg = buffer[: conj.start()].strip()
            pre_count = len(pre_seg.split())
            post_words = len(buffer[conj.start() :].split())
            if 6 <= pre_count <= 9 and post_words >= 2:
                return conj.start() + 1

    # 3. Extreme run-on fallback with zero punctuation (>= 22 words with no commas or periods)
    if word_count >= 22:
        conj_pattern = r"(\s+(?:" + "|".join(CLAUSE_CONJUNCTIONS) + r")\b)"
        conj_matches = list(re.finditer(conj_pattern, buffer, flags=re.IGNORECASE))
        if conj_matches:
            for cm in conj_matches:
                pre_seg = buffer[: cm.start()].strip()
                if len(pre_seg.split()) >= 10:
                    return cm.start() + 1

    return -1


EXPLANATORY_KEYWORDS = {
    "relativity",
    "quantum",
    "physics",
    "equation",
    "formula",
    "theorem",
    "architecture",
    "algorithm",
    "mechanism",
    "hypothesis",
    "principle",
    "thermodynamics",
    "philosophy",
    "electromagnetism",
    "gravitational",
    "differentiation",
    "integration",
    "astrophysics",
    "neuroscience",
}


def determine_sentence_pacing(sentence: str) -> float:
    """
    Calculates dynamic speech speed multiplier (0.85x to 1.2x) based on sentence length,
    clause complexity, and technical/explanatory content.
    """
    words = sentence.split()
    word_count = len(words)
    if word_count == 0:
        return 1.0

    lower_sentence = sentence.lower()
    has_technical = any(kw in lower_sentence for kw in EXPLANATORY_KEYWORDS)

    # 1. Very short sentences (e.g. "Alarm set for 7:00 AM, sir.") -> Snappy pace (1.20x)
    if word_count < 12 and not has_technical:
        return 1.20

    # 2. Medium short sentences -> Slightly faster than normal (1.10x)
    if word_count < 22 and not has_technical:
        return 1.10

    # 3. Technical or explanatory sentences -> Deliberate, clear pace (0.90x - 0.95x)
    if has_technical:
        return 0.90 if word_count > 25 else 0.95

    # 4. Long descriptive sentences -> Clear, measured pace (0.95x - 1.0x)
    if word_count >= 30:
        return 0.95

    return 1.0


class TextToSpeech:
    """
    Decoupled general TTS Audio Playback & Streaming Engine.
    Handles text cleaning, full sentence prosody, dynamic pacing, and background playback.
    """

    def __init__(
        self, provider: BaseTTSProvider | None = None, sample_rate: int = 24000
    ):
        t0 = perf_counter()
        if provider is None:
            self.provider = get_tts_provider()
        else:
            self.provider = provider

        logger.info(
            f"[TTS Engine] Initialized using provider: {self.provider.__class__.__name__} in {perf_counter() - t0:.2f}s"
        )

        self.sample_rate = getattr(self.provider, "sample_rate", sample_rate)
        self.interrupt_event = threading.Event()
        self.current_generation_id = 0
        self.generation_lock = threading.Lock()

        # Thread-safe queue to pass (audio_data, gen_id) to the sounddevice output stream
        self.audio_queue = queue.Queue()
        self.remainder = None
        self.remainder_gen_id = None
        self.starvation_start = None

        # Playback jitter-buffering state
        self.playback_started_gen_id = -1
        self.first_audio_arrival_time = None

        # Queue for phrases waiting to be synthesized by worker: stores (text, speed, gen_id) tuples
        self.tts_queue = queue.Queue()

        # Dedicated background worker for provider synthesis
        self.worker_thread = threading.Thread(
            target=self._synthesis_worker, daemon=True
        )
        self.worker_thread.start()

        # Non-blocking background audio output stream
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            dtype="float32",
        )
        self.stream.start()

    def interrupt(self):
        """Immediately stops all speech output, increments generation epoch, and flushes queues."""
        with self.generation_lock:
            self.current_generation_id += 1
            self.interrupt_event.set()

            # 1. Drain TTS sentence queue
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                    self.tts_queue.task_done()
                except Exception:
                    break

            # 2. Drain Audio chunk queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except Exception:
                    break

            # 3. Discard leftover playback buffer
            self.remainder = None
            self.remainder_gen_id = None
            self.playback_started_gen_id = -1
            self.first_audio_arrival_time = None

        logger.info(
            f"[TTS] Interrupted! Epoch advanced to {self.current_generation_id} and queues flushed."
        )

    def is_interrupted(self) -> bool:
        return self.interrupt_event.is_set()

    def reset_interrupt(self):
        self.interrupt_event.clear()

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
                if item is not None:
                    data = item[0] if isinstance(item, tuple) else item
                    if hasattr(data, "__len__"):
                        samples += len(data)
        return samples / float(self.sample_rate)

    def _synthesis_worker(self):
        """Single worker thread that consumes text phrases and synthesizes audio via provider."""
        while True:
            try:
                item = self.tts_queue.get()
                if item is None:
                    break

                if isinstance(item, tuple) and len(item) == 3:
                    text, speed, gen_id = item
                elif isinstance(item, tuple) and len(item) == 2:
                    text, speed = item
                    gen_id = self.current_generation_id
                else:
                    text, speed, gen_id = item, 1.0, self.current_generation_id

                if (
                    gen_id != self.current_generation_id
                    or self.interrupt_event.is_set()
                ):
                    logger.info(
                        f"[Worker] Skipping stale task for gen {gen_id} (current={self.current_generation_id})"
                    )
                    self.tts_queue.task_done()
                    continue

                logger.info(f"WORKER <<< {text} (speed={speed:.2f}x, gen={gen_id})")
                self._synthesize_to_queue(text, speed=speed, gen_id=gen_id)
                self.tts_queue.task_done()
            except Exception as e:
                logger.error(f"[Worker Error] {e}")

    def _audio_callback(self, outdata, _, __, status):
        """Continuously feeds the sound card from the audio queue."""
        if status:
            logger.info(f"[Audio Status Warning] {status}")

        if self.interrupt_event.is_set():
            outdata.fill(0)
            return

        needed = len(outdata)
        out_buf = np.zeros(needed, dtype=np.float32)
        filled = 0

        # Initial jitter buffer gate for newly started generation:
        # Ensures a brief cushion before feeding the soundcard, so CPU synthesis stays ahead of playback.
        if self.playback_started_gen_id != self.current_generation_id:
            buffered_sec = self.get_buffered_duration()
            if buffered_sec <= 0:
                outdata.fill(0)
                return

            if self.first_audio_arrival_time is None:
                self.first_audio_arrival_time = time.time()

            elapsed_wait = time.time() - self.first_audio_arrival_time
            # Adaptive jitter buffer gate:
            # 1. If tts_queue is empty (single-sentence response), start immediately with 0 delay.
            # 2. If multi-phrase response is queued, require at least 1.8s of audio (or 0.60s max wait)
            #    so CPU synthesis maintains an unbroken lead ahead of playback.
            should_start = (
                self.tts_queue.empty()
                or buffered_sec >= 1.8
                or elapsed_wait >= 0.60
            )
            if should_start:
                self.playback_started_gen_id = self.current_generation_id
                self.first_audio_arrival_time = None
            else:
                outdata.fill(0)
                return

        # 1. Use leftover audio from previous callback (validate generation)
        if self.remainder is not None and len(self.remainder) > 0:
            if (
                self.remainder_gen_id == self.current_generation_id
                and not self.interrupt_event.is_set()
            ):
                take = min(needed - filled, len(self.remainder))
                out_buf[filled : filled + take] = self.remainder[:take]
                self.remainder = self.remainder[take:]
                filled += take
            else:
                self.remainder = None
                self.remainder_gen_id = None

        # 2. Get audio chunks from queue
        while filled < needed:
            if self.interrupt_event.is_set():
                outdata.fill(0)
                return

            try:
                item = self.audio_queue.get_nowait()
            except queue.Empty:
                break

            if isinstance(item, tuple):
                data, gen_id = item
            else:
                data, gen_id = item, self.current_generation_id

            # Discard any audio that belongs to an older/interrupted generation epoch
            if gen_id != self.current_generation_id or self.interrupt_event.is_set():
                continue

            if self.starvation_start is not None:
                duration = time.time() - self.starvation_start
                logger.info(
                    f"[AUDIO] Audio resumed. Starved for {duration:.3f} seconds"
                )
                self.starvation_start = None

            take = min(needed - filled, len(data))
            out_buf[filled : filled + take] = data[:take]
            filled += take

            if take < len(data):
                self.remainder = data[take:]
                self.remainder_gen_id = gen_id
                break

        if filled == 0:
            outdata.fill(0)
            # Only report starvation if synthesis is still pending (more work in tts_queue).
            # Once speech has legitimately completed, returning to silence is normal.
            if self.starvation_start is None and not self.tts_queue.empty():
                self.starvation_start = time.time()
                logger.warning("[AUDIO] Queue starved")
        else:
            if filled < needed:
                out_buf[filled:] = 0
            outdata[:] = out_buf.reshape(-1, 1)

    def speak_stream(self, llm_generator) -> None:
        """
        Consumes tokens from the LLM generator.
        Streams complete natural sentences to the synthesizer with dynamic pacing.
        Locks pacing to 0.95x throughout explanations for smooth, deliberate speech.
        """
        raw_buffer = ""
        is_explanation_mode = False
        is_first_chunk = True

        with self.generation_lock:
            self.current_generation_id += 1
            gen_id = self.current_generation_id
            self.playback_started_gen_id = -1
            self.first_audio_arrival_time = None
            self.reset_interrupt()

        logger.info(f"[TTS] Starting speak_stream for generation {gen_id}")

        for text_chunk, _ in llm_generator:
            if gen_id != self.current_generation_id or self.interrupt_event.is_set():
                logger.info(
                    f"[TTS] Interrupted during LLM token stream for gen {gen_id}."
                )
                break

            logger.info(f"Received: {text_chunk!r}")
            raw_buffer += text_chunk

            while True:
                if (
                    gen_id != self.current_generation_id
                    or self.interrupt_event.is_set()
                ):
                    break

                buffered_sec = self.get_buffered_duration()
                split_idx = find_sentence_split_point(
                    raw_buffer,
                    is_first_chunk=is_first_chunk,
                    buffered_duration=buffered_sec,
                )

                if split_idx == -1:
                    break

                sentence_raw = raw_buffer[:split_idx]
                raw_buffer = raw_buffer[split_idx:]

                clean_sentence = clean_text_for_tts(sentence_raw)
                if clean_sentence and any(c.isalnum() for c in clean_sentence):
                    is_first_chunk = False
                    sentence_pacing = determine_sentence_pacing(clean_sentence)

                    if sentence_pacing <= 0.95 or any(
                        kw in clean_sentence.lower() for kw in EXPLANATORY_KEYWORDS
                    ):
                        is_explanation_mode = True

                    speed = 0.95 if is_explanation_mode else sentence_pacing
                    logger.info(
                        f"QUEUE >>> {clean_sentence} (speed={speed:.2f}x, gen={gen_id})"
                    )
                    self.tts_queue.put((clean_sentence, speed, gen_id))

        if gen_id != self.current_generation_id or self.interrupt_event.is_set():
            return

        # Handle any leftover trailing text at end of generation
        if raw_buffer.strip():
            clean_sentence = clean_text_for_tts(raw_buffer)
            if clean_sentence and any(c.isalnum() for c in clean_sentence):
                sentence_pacing = determine_sentence_pacing(clean_sentence)
                if sentence_pacing <= 0.95 or any(
                    kw in clean_sentence.lower() for kw in EXPLANATORY_KEYWORDS
                ):
                    is_explanation_mode = True
                speed = 0.95 if is_explanation_mode else sentence_pacing
                logger.info(
                    f"QUEUE (FINAL) >>> {clean_sentence} (speed={speed:.2f}x, gen={gen_id})"
                )
                self.tts_queue.put((clean_sentence, speed, gen_id))

    def _synthesize_to_queue(
        self, text: str, speed: float = 1.0, gen_id: int | None = None
    ):
        """Delegates synthesis to provider and pushes float32 PCM audio chunks to playback queue."""
        if gen_id is None:
            gen_id = self.current_generation_id

        logger.info(f"[SYNTH] START: {text!r} (speed={speed:.2f}x, gen={gen_id})")
        try:
            t0 = time.time()
            chunk_count = 0
            first_audio = True

            for audio_data in self.provider.synthesize(text, speed=speed):
                if (
                    gen_id != self.current_generation_id
                    or self.interrupt_event.is_set()
                ):
                    logger.info(
                        f"[SYNTH] Stale synthesis discarded for gen {gen_id} (current={self.current_generation_id})."
                    )
                    break

                chunk_count += 1
                if first_audio:
                    logger.info(
                        f"[SYNTH] FIRST AUDIO CHUNK after {time.time() - t0:.3f}s (gen={gen_id})"
                    )
                    first_audio = False

                if (
                    audio_data is not None
                    and len(audio_data) > 0
                    and gen_id == self.current_generation_id
                    and not self.interrupt_event.is_set()
                ):
                    logger.info(
                        f"[SYNTH] Queueing chunk #{chunk_count} ({len(audio_data)} samples, gen={gen_id})"
                    )
                    self.audio_queue.put((audio_data, gen_id))

            logger.info(
                f"[SYNTH] DONE after {time.time() - t0:.3f}s ({chunk_count} chunks, gen={gen_id})"
            )

        except Exception as e:
            logger.exception(f"[SYNTH ERROR] {e}")

    def speak_text(self, text: str, speed: float = 1.0) -> None:
        """Queues a plain text string to be synthesized and spoken out loud via TTS."""
        clean = clean_text_for_tts(text)
        if clean and any(c.isalnum() for c in clean):
            with self.generation_lock:
                self.current_generation_id += 1
                gen_id = self.current_generation_id
                self.playback_started_gen_id = -1
                self.first_audio_arrival_time = None
            logger.info(
                f"[TTS] Direct speak text queued: '{clean}' (speed={speed:.2f}x, gen={gen_id})"
            )
            self.tts_queue.put((clean, speed, gen_id))

    def close(self):
        """Stops the audio output stream cleanly when shutting down."""
        self.stream.stop()
        self.stream.close()
