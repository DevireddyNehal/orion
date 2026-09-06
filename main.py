from dotenv import load_dotenv
load_dotenv(override=True)

# ==========================================
# FORCE PYTORCH TO BASS-BOOST CPU ONLY (NO CUDA TOUCH)
import sys
from numpy import rint
import torch

# Overwrite the deadlocking C++ functions with fake ones instantly
torch._C._cuda_getDeviceCount = lambda: 0
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import re
import time
from loguru import logger
from voice.recorder import AudioRecorder
from voice.stt import WhisperSTT
from llm.llm import LLMOrchestrator
from voice.tts import TextToSpeech
from tools.reminders import reminder_manager

WHISPER_SILENCE_ARTIFACTS = {
    "", "thank you", "thanks for watching", "you", "blank_audio",
    "subtitles by", "bye", "silence", "thank you so much", "thanks"
}

def main():
    # 1. Initialize the Recorder FIRST so it gets clean, un-hijacked access to sounddevice
    t = time.time()
    recorder = AudioRecorder()
    logger.info("Recorder init time: {}s", time.time() - t)

    # 2. Initialize the heavy model backends second
    t = time.time()
    stt = WhisperSTT()
    logger.info("STT init time: {}s", time.time() - t)

    t = time.time()
    llm_orchestrator = LLMOrchestrator()
    logger.info("LLM Orchestrator init time: {}s", time.time() - t)

    t = time.time()
    tts = TextToSpeech()
    logger.info("TTS init time: {}s", time.time() - t)

    # Connect instant interrupt / barge-in trigger to TTS engine
    recorder.register_on_press_callback(tts.interrupt)

    # Connect TTS engine to internal reminder manager
    reminder_manager.set_tts(tts)

    while True:
        try:
            logger.info("Waiting for recorder...")
            pcm_data, record_duration = recorder.record()
            t_speech_end = time.time()
            logger.info("Returned from recorder | Record duration: {:.2f}s", record_duration)

            if not pcm_data:
                continue

            # Pre-warm TTS pipeline asynchronously while STT and LLM run
            tts.warmup()

            logger.info("Transcribing...")
            t_stt_start = time.time()
            text = stt.transcribe(pcm_data)
            stt_time = time.time() - t_stt_start

            # Accidental press filter: if no words or only Whisper silence artifacts, silently skip
            cleaned = re.sub(r'[^\w\s]', '', text or '').strip().lower()
            if not cleaned or cleaned in WHISPER_SILENCE_ARTIFACTS:
                continue

            logger.info("STT Time: {:.3f}s | Result: '{}'", stt_time, text)

            # Stream LLM generation with native tool support directly to TTS
            logger.info("AI starting response pipeline...")
            t_pipe_start = time.time()
            stream = llm_orchestrator.generate_response_stream(text)
            tts.speak_stream(stream)
            total_turn_time = time.time() - t_speech_end
            logger.info("Response stream finished in {:.3f}s (Total turn time from key release: {:.3f}s)",
                        time.time() - t_pipe_start, total_turn_time)

            print() # Inserts a clean newline after streaming text finishes
        
        except KeyboardInterrupt:
            logger.info("Shutting down cleanly...")
            tts.close() # Safely releases the sound card
            break

if __name__ == "__main__":
    main()