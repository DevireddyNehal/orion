import queue
import time
import threading
import numpy as np
import sounddevice as sd
from kokoro import KPipeline
from time import perf_counter
from loguru import logger

class TextToSpeech:
    def __init__(self, voice: str = "af_heart", lang_code: str = "a"):
        logger.info("[TTS] Initializing Kokoro Zero-Latency Pipeline...")
        t0 = perf_counter()
        self.pipeline = KPipeline(lang_code=lang_code,
            repo_id="hexgrad/Kokoro-82M",
            device="cpu"
        )
        print(f"KPipeline init: {perf_counter() - t0:.2f}s")
        
        # Warm up the pipeline to avoid 80s delay on first utterance
        # This forces model loading, ONNX session creation, and voice loading
        logger.info("[TTS] Warming up pipeline (this may take a few seconds)...")
        try:
            # Consume the generator completely to force full initialization
            list(self.pipeline(" ", voice=voice))
            logger.info("[TTS] Pipeline warmed up successfully.")
        except Exception as e:
            logger.error(f"[TTS] Warm-up failed: {e}")
            
        self.voice = voice
        self.sample_rate = 24000      
        # Thread-safe queue to pass audio chunks to the playback stream
        self.audio_queue = queue.Queue()
        self.starvation_start = None
        
        # Queue for phrases waiting to be synthesized
        self.tts_queue = queue.Queue()
        
        # Start a single dedicated background worker for synthesis
        self.worker_thread = threading.Thread(target=self._synthesis_worker, daemon=True)
        self.worker_thread.start()
        
        # Start the non-blocking background audio output stream
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate, 
            channels=1, 
            callback=self._audio_callback,
            dtype='float32'
        )
        self.stream.start()

    def _synthesis_worker(self):
        """Single worker thread that consumes text phrases and synthesizes audio."""
        while True:
            try:
                text = self.tts_queue.get()
                logger.info(f"WORKER <<< {text}")
                if text is None:
                    break
                self._synthesize_to_queue(text)
                self.tts_queue.task_done()
            except Exception as e:
                logger.error(f"[Worker Error] {e}")

    def _audio_callback(self, outdata, _, __, status):
        """Continuously feeds the sound card from the audio queue."""
        if status:
            logger.info(f"[Audio Status Warning] {status}")
        
        try:
            # Try to get enough audio samples to fill the sound buffer
            data = self.audio_queue.get_nowait()
            
            if self.starvation_start is not None:
                duration = time.time() - self.starvation_start
                logger.info(f"[AUDIO] Audio resumed. Starved for {duration:.3f} seconds")
                self.starvation_start = None

            # If the chunk is smaller than the requested buffer, pad it with zeros
            if len(data) < len(outdata):
                outdata[:len(data)] = data.reshape(-1, 1)
                outdata[len(data):] = 0
            else:
                outdata[:] = data[:len(outdata)].reshape(-1, 1)
                # Put the remaining audio samples back in the front of the queue
                if len(data) > len(outdata):
                    self.audio_queue.queue.appendleft(data[len(outdata):])
        except queue.Empty:
            # If no audio is ready, output absolute silence instead of stuttering
            outdata.fill(0)
            if self.starvation_start is None:
                self.starvation_start = time.time()
                logger.warning("[AUDIO] Queue starved")
    
    def speak_stream(self, llm_generator) -> None:
        """
        Consumes the LLM generator and enqueues phrases for the synthesis worker.
        """
        sentence_buffer = ""

        for text_chunk, _ in llm_generator:
            logger.info(f"Received: {repr(text_chunk)}")
            logger.info(text_chunk)
            sentence_buffer += text_chunk

            # Send to Kokoro as soon as punctuation is detected
            if (any(p in sentence_buffer for p in [".", ",", ";", "!", "?", "\n"])
                or (
                    len(sentence_buffer) >= 60
                    and sentence_buffer.endswith(" ")
                )
            ):
                clean_phrase = sentence_buffer.strip()
                if clean_phrase:
                    logger.info(f"QUEUE >>> {clean_phrase}")
                    # Enqueue phrase for the dedicated worker to preserve FIFO order
                    self.tts_queue.put(clean_phrase)
                sentence_buffer = ""

        # Handle any leftover text trailing at the end
        if sentence_buffer.strip():
            self.tts_queue.put(sentence_buffer.strip())

    def _synthesize_to_queue(self, text: str):
        """Runs Kokoro synthesis and pushes audio chunks to the playback queue."""

        logger.info(f"[SYNTH] START: {repr(text)}")

        try:
            t0 = time.time()

            logger.info("[SYNTH] Calling pipeline()...")
            audio_generator = self.pipeline(text, voice=self.voice)
            logger.info(f"[SYNTH] pipeline() returned generator in {time.time() - t0:.3f}s")

            first_audio = True
            chunk_count = 0

            for _, _, audio in audio_generator:
                chunk_count += 1

                if first_audio:
                    logger.info(
                        f"[SYNTH] FIRST AUDIO CHUNK after {time.time() - t0:.3f}s"
                    )
                    first_audio = False

                if audio is None or len(audio) == 0:
                    continue

                audio_data = np.asarray(audio, dtype=np.float32)

                logger.info(
                    f"[SYNTH] Queueing chunk #{chunk_count} ({len(audio_data)} samples)"
                )

                self.audio_queue.put(audio_data)

            logger.info(
                f"[SYNTH] DONE after {time.time() - t0:.3f}s "
                f"({chunk_count} chunks)"
            )

        except Exception as e:
            logger.exception(f"[SYNTH ERROR] {e}")

    def close(self):
        """Stops the audio device stream cleanly when closing the app."""
        self.stream.stop()
        self.stream.close()