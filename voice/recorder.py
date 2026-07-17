from threading import Event

import time
import loguru
import numpy as np
import sounddevice as sd
from pynput import keyboard

logger = loguru.logger


class AudioRecorder:
    def __init__(self, samplerate=16000, channels=1):
        self.samplerate = samplerate
        self.channels = channels

        self.key = keyboard.Key.ctrl_r
        self.record_event = Event()

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )
        self.listener.start()

    def on_press(self, key):
        if key == self.key and not self.record_event.is_set():
            logger.info("🎤 Recording started")
            self.record_event.set()

    def on_release(self, key):
        if key == self.key and self.record_event.is_set():
            logger.info("🛑 Recording stopped")
            self.record_event.clear()

    def record(self) -> bytes:
        chunks = []

        def callback(indata, frames, time, status):
            if status:
                logger.warning(status)
            chunks.append(indata.copy())

        print(f"Hold {self.key} to record...")

        # Wait until the key is pressed
        self.record_event.wait()

        print("Recording...")

        with sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="int16",
            callback=callback,
        ):
            # Stay recording while the key is held
            while self.record_event.is_set():
                sd.sleep(20)

        print("Recording stopped.")

        if not chunks:
            return b""

        audio = np.concatenate(chunks, axis=0)
        return audio.tobytes()