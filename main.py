# ==========================================
# FORCE PYTORCH TO BASS-BOOST CPU ONLY (NO CUDA TOUCH)
import sys
from numpy import rint
import torch

# Overwrite the deadlocking C++ functions with fake ones instantly
torch._C._cuda_getDeviceCount = lambda: 0
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

from loguru import logger
from voice.recorder import AudioRecorder
from voice.stt import WhisperSTT
from llm.llm import LLMOrchestrator
from voice.tts import TextToSpeech
import time

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

    while True:
        try:
            t = time.time()

            logger.info("Waiting for recorder...")
            pcm_data = recorder.record()
            logger.info("Returned from recorder")
            logger.info("Recording...")
            logger.info("Record: {}", time.time() - t)

            if not pcm_data:
                continue

            logger.info("Transcribing...")

            t = time.time()
            text = stt.transcribe(pcm_data)
            route = llm_orchestrator.intent_classifier(text)

            logger.info("Route model: {}", route.model)
            logger.info("Capabilities: {}", route.capabilities)
            logger.info("STT Time: {}", time.time() - t)
            logger.info("Result: {}", text)

            if text is not None and text.strip() != "":
                # 1. Grab the generator object stream ONCE
                stream = llm_orchestrator.generate_response_stream(
                    text,
                    route.model,
                    prompt=llm_orchestrator.response_system_prompt
                ) 
                        
                # 2. Hand the entire stream over to Kokoro
                logger.info("AI starting response pipeline...")
                tts.speak_stream(stream)

                print() # Inserts a clean newline after streaming text finishes
            
        except KeyboardInterrupt:
            logger.info("Shutting down cleanly...")
            tts.close() # Safely releases the sound card
            break

if __name__ == "__main__":
    main()