"""Always-on local Vosk listener with a wake-phrase gate."""
import json
import queue
import threading
from pathlib import Path

import sounddevice as sd
from vosk import KaldiRecognizer, Model


class WakeWordListener:
    def __init__(self, model_path: str | Path, wake_words=("hey glasses", "glasses")):
        self.model = Model(str(model_path))
        self.wake_words = tuple(wake_words)
        # Cap retained audio so a slow text-model response cannot accumulate
        # unbounded memory while the camera microphone continues recording.
        self._audio: queue.Queue[bytes] = queue.Queue(maxsize=32)
        self._stop = threading.Event()
        self._muted = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self.last_partial = ""

    def mute(self) -> None:
        self._muted.set()
        self._clear_audio()

    def unmute(self) -> None:
        self._clear_audio()
        self._muted.clear()

    def _clear_audio(self) -> None:
        while True:
            try:
                self._audio.get_nowait()
            except queue.Empty:
                return

    def start(self, on_command) -> None:
        self._thread = threading.Thread(target=self._run, args=(on_command,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _callback(self, indata, frames, time_info, status) -> None:
        if self._muted.is_set():
            return
        if status:
            print(f"[mic] {status}", flush=True)
        else:
            try:
                self._audio.put_nowait(bytes(indata))
            except queue.Full:
                pass

    def _run(self, on_command) -> None:
        recognizer = KaldiRecognizer(self.model, 16000)
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                               channels=1, device=1, callback=self._callback):
            while not self._stop.is_set():
                try:
                    chunk = self._audio.get(timeout=0.2)
                except queue.Empty:
                    continue
                if not recognizer.AcceptWaveform(chunk):
                    partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                    if partial and partial != self.last_partial:
                        self.last_partial = partial
                        print(f"[mic hearing] {partial}", flush=True)
                    continue
                text = json.loads(recognizer.Result()).get("text", "").lower().strip()
                if text:
                    print(f"[mic final] {text}", flush=True)
                if not text:
                    continue
                if not self._active:
                    matched = next((w for w in self.wake_words if w in text), None)
                    if matched is None:
                        continue
                    text = text.split(matched, 1)[1].strip(" ,.!?")
                    self._active = True
                if text:
                    self._active = False
                    threading.Thread(
                        target=on_command,
                        args=(text,),
                        daemon=True,
                    ).start()
                recognizer.Reset()
