import selectors
import threading
import time
from threading import Event

import loguru
import numpy as np
import sounddevice as sd
from evdev import InputDevice, categorize, ecodes, list_devices

logger = loguru.logger


class AudioRecorder:

    def __init__(self, samplerate=16000, channels=1):
        self.samplerate = samplerate
        self.channels = channels

        self.target_key_code = ecodes.KEY_RIGHTCTRL
        self.record_event = Event()

        self.device_paths = self._discover_keyboards()

        self.listener_thread = threading.Thread(
            target=self._evdev_listener_loop, daemon=True
        )
        self.listener_thread.start()


    def record(self) -> bytes:
        chunks = []

        def callback(indata, frames, time, status):
            if status:
                logger.warning(status)
            chunks.append(indata.copy())

        print("Hold RIGHT CTRL key to record...")

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

    def _discover_keyboards(self) -> list:
        """Universally scans and locates all active keyboard interfaces on the host system."""
        found_devices = []
        for path in list_devices():
            try:
                dev = InputDevice(path)
                capabilities = dev.capabilities()
                # Check if device supports key events (EV_KEY)
                if ecodes.EV_KEY in capabilities:
                    supported_keys = capabilities[ecodes.EV_KEY]
                    # Ensure it supports standard alphanumeric/modifier layout keys
                    if (
                        ecodes.KEY_RIGHTCTRL in supported_keys
                        and ecodes.KEY_A in supported_keys
                    ):
                        found_devices.append(path)
            except (PermissionError, FileNotFoundError):
                continue

        if not found_devices:
            logger.warning(
                "No hardware keyboards matched filter criteria. Falling back to global search."
            )
            return (
                ["/dev/input/event4"]
                if "/dev/input/event4" in list_devices()
                else []
            )

        logger.info(
            f"Universal discovery engine found {len(found_devices)} keyboard(s): {found_devices}"
        )
        return found_devices

    def _evdev_listener_loop(self):
        selector = selectors.DefaultSelector()

        # Connect tracking to every keyboard interface dynamically discovered
        for path in self.device_paths:
            try:
                device = InputDevice(path)
                selector.register(device, selectors.EVENT_READ)
                logger.info(f"Successfully listening to device: {device.name}")
            except Exception as e:
                logger.debug(
                    f"Bypassing locked or unreadable device path {path}: {e}"
                )

        if not selector.get_map():
            logger.critical(
                "Fatal: No accessible keyboard streams available. Check user permissions."
            )
            return

        while True:
            try:
                for key, _ in selector.select(timeout=None):
                    device = key.fileobj
                    # Exhaust all events pending in buffer pipeline
                    try:
                        events = list(device.read())
                    except (BlockingIOError, OSError):
                        continue

                    for event in events:
                        if event.type == ecodes.EV_KEY:
                            key_event = categorize(event)
                            if key_event.scancode == self.target_key_code:
                                if key_event.keystate == 1:  # Key Down
                                    logger.info(
                                        f"🎯 Global press intercepted from: {device.name}"
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
