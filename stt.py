"""Always-on local Vosk listener with a wake-phrase gate."""
import json
import queue
import re
import threading
import time
from pathlib import Path

import sounddevice as sd
from vosk import KaldiRecognizer, Model


class WakeWordListener:
    def __init__(self, model_path: str | Path, wake_words=("pupil",)):
        self.model = Model(str(model_path))
        self.wake_words = tuple(wake_words)
        self.wake_aliases = (
            "pupil", "people", "pew poll", "pewpol", "pew pull", "pup pill",
            "pupul", "purple", "kubo", "q ball", "cue ball"
        )
        # Cap retained audio so a slow text-model response cannot accumulate
        # unbounded memory while the camera microphone continues recording.
        self._audio: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._capture_reply: queue.Queue[str] = queue.Queue(maxsize=1)
        self._capture_until = 0.0
        self._capture_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False

    def start(self, on_command) -> None:
        self._thread = threading.Thread(target=self._run, args=(on_command,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _wake_match(self, text: str) -> str | None:
        return "pupil" if "pupil" in self._correct_wake_word(text) else None

    def _correct_wake_word(self, text: str) -> str:
        corrected = text
        for alias in sorted(self.wake_aliases, key=len, reverse=True):
            corrected = re.sub(rf"\b{re.escape(alias)}\b", "pupil", corrected)
        return corrected

    def request_phrase(self, timeout: float = 10.0) -> str | None:
        while not self._capture_reply.empty():
            try:
                self._capture_reply.get_nowait()
            except queue.Empty:
                break
        with self._capture_lock:
            self._capture_until = time.monotonic() + timeout
        try:
            return self._capture_reply.get(timeout=timeout)
        except queue.Empty:
            return None
        finally:
            with self._capture_lock:
                self._capture_until = 0.0

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[mic] {status}", flush=True)
        try:
            self._audio.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def _run(self, on_command) -> None:
        recognizer = KaldiRecognizer(self.model, 16000)
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                               channels=1, callback=self._callback):
            while not self._stop.is_set():
                try:
                    chunk = self._audio.get(timeout=0.2)
                except queue.Empty:
                    continue
                if not recognizer.AcceptWaveform(chunk):
                    continue
                text = json.loads(recognizer.Result()).get("text", "").lower().strip()
                text = self._correct_wake_word(text)
                if not text:
                    continue
                speech_end = time.monotonic()
                with self._capture_lock:
                    capturing = self._capture_until and speech_end <= self._capture_until
                if capturing:
                    try:
                        self._capture_reply.put_nowait(text)
                    except queue.Full:
                        pass
                    recognizer.Reset()
                    continue
                if not self._active:
                    matched = self._wake_match(text)
                    if matched is None:
                        continue
                    text = text.split(matched, 1)[1].strip(" ,.!?")
                    self._active = True
                if text:
                    threading.Thread(
                        target=on_command,
                        args=(text, speech_end),
                        daemon=True,
                    ).start()
                    self._active = False
                recognizer.Reset()
