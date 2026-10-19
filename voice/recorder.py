import os
import selectors
import threading
import time
from collections.abc import Callable
from threading import Event

import loguru
import numpy as np
import sounddevice as sd
from evdev import InputDevice, KeyEvent, categorize, ecodes, list_devices

logger = loguru.logger


class AudioRecorder:
    def __init__(self, samplerate=16000, channels=1):
        self.samplerate = samplerate
        self.channels = channels

        self.target_key_name = os.getenv("RECORD_KEY", "KEY_RIGHTCTRL").strip().upper()
        if hasattr(ecodes, self.target_key_name):
            self.target_key_code = getattr(ecodes, self.target_key_name)
        else:
            logger.warning(
                f"[AudioRecorder] Unknown RECORD_KEY '{self.target_key_name}'. Defaulting to KEY_RIGHTCTRL."
            )
            self.target_key_name = "KEY_RIGHTCTRL"
            self.target_key_code = ecodes.KEY_RIGHTCTRL

        self.record_event = Event()
        self.on_press_callbacks: list[Callable[[], None]] = []

        self.device_paths = self._discover_keyboards()

        self.listener_thread = threading.Thread(
            target=self._evdev_listener_loop, daemon=True
        )
        self.listener_thread.start()

    def register_on_press_callback(self, callback: Callable[[], None]):
        """Registers a callback function to be called immediately when the record/interrupt key is pressed."""
        if callback not in self.on_press_callbacks:
            self.on_press_callbacks.append(callback)

    def record(self) -> tuple[bytes, float]:
        chunks = []

        def callback(indata, frames, time, status):
            if status:
                logger.warning(status)
            chunks.append(indata.copy())

        display_key = self.target_key_name.replace("KEY_", "").replace("_", " ")
        print(f"Hold {display_key} key to record...")

        # Wait until the key is pressed
        self.record_event.wait()
        t_rec_start = time.time()

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

        t_rec_end = time.time()
        record_duration = t_rec_end - t_rec_start
        print("Recording stopped.")

        if not chunks:
            return b"", record_duration

        audio = np.concatenate(chunks, axis=0)
        return audio.tobytes(), record_duration

    def _discover_keyboards(self) -> list:
        """Universally scans and locates all active keyboard interfaces on the host system."""
        # 1. Check for user-specified device path
        explicit_device = os.getenv("RECORD_DEVICE_PATH", "").strip()
        if explicit_device:
            if explicit_device in list_devices():
                logger.info(
                    f"[AudioRecorder] Using configured RECORD_DEVICE_PATH: {explicit_device}"
                )
                return [explicit_device]
            logger.warning(
                f"[AudioRecorder] Configured RECORD_DEVICE_PATH '{explicit_device}' not found in available devices."
            )

        found_devices = []
        fallback_keyboards = []
        for path in list_devices():
            try:
                dev = InputDevice(path)
                capabilities = dev.capabilities()
                if ecodes.EV_KEY in capabilities:
                    supported_keys = capabilities[ecodes.EV_KEY]
                    # Check if it supports the target record key and standard alphanumeric keys
                    if (
                        self.target_key_code in supported_keys
                        and ecodes.KEY_A in supported_keys
                    ):
                        found_devices.append(path)
                    elif (
                        ecodes.KEY_A in supported_keys
                        or ecodes.KEY_SPACE in supported_keys
                    ):
                        fallback_keyboards.append(path)
            except (PermissionError, FileNotFoundError):
                continue

        if found_devices:
            logger.info(
                f"Universal discovery engine found {len(found_devices)} keyboard(s): {found_devices}"
            )
            return found_devices

        if fallback_keyboards:
            logger.warning(
                f"No devices explicitly advertised {self.target_key_name}. Falling back to general keyboard interface(s): {fallback_keyboards}"
            )
            return fallback_keyboards

        logger.warning(
            "No hardware keyboards matched filter criteria. Falling back to global search."
        )
        return ["/dev/input/event4"] if "/dev/input/event4" in list_devices() else []

    def _evdev_listener_loop(self):
        selector = selectors.DefaultSelector()

        # Connect tracking to every keyboard interface dynamically discovered
        for path in self.device_paths:
            try:
                device = InputDevice(path)
                selector.register(device, selectors.EVENT_READ)
                logger.info(f"Successfully listening to device: {device.name}")
            except Exception as e:
                logger.debug(f"Bypassing locked or unreadable device path {path}: {e}")

        if not selector.get_map():
            logger.critical(
                "Fatal: No accessible keyboard streams available. Check user permissions."
            )
            return

        while True:
            try:
                for key, _ in selector.select(timeout=None):
                    if not isinstance(key.fileobj, InputDevice):
                        continue
                    device = key.fileobj
                    # Exhaust all events pending in buffer pipeline
                    try:
                        events = list(device.read())
                    except (BlockingIOError, OSError):
                        continue

                    for event in events:
                        if event.type == ecodes.EV_KEY:
                            key_event = categorize(event)
                            if not isinstance(key_event, KeyEvent):
                                continue

                            # Push-to-Talk Barge-In trigger (RIGHT CTRL)
                            if key_event.scancode == self.target_key_code:
                                if key_event.keystate == 1:  # Key Down
                                    logger.info(
                                        f"🎯 Global press intercepted from: {device.name}"
                                    )
                                    # Trigger instant interrupt of TTS/audio
                                    for cb in self.on_press_callbacks:
                                        try:
                                            cb()
                                        except Exception as cb_err:
                                            logger.error(
                                                f"Callback error on press: {cb_err}"
                                            )

                                    if not self.record_event.is_set():
                                        self.record_event.set()
                                elif key_event.keystate == 0:  # Key Up
                                    logger.info(
                                        f"🏁 Global release intercepted from: {device.name}"
                                    )
                                    if self.record_event.is_set():
                                        self.record_event.clear()
            except Exception as e:
                logger.error(f"Event pipeline exception: {e}")
                time.sleep(1)
