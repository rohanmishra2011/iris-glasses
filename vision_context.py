"""Offline room inference using scanned rooms first and Qwen fallback if needed."""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.request

from config import Tuning
from room_memory import RoomMemory


class RoomClassifierWorker:
    def __init__(
        self,
        state,
        room_memory: RoomMemory,
        tuning: Tuning,
        model: str = "qwen2.5:0.5b-instruct",
        interval: float = 30.0,
    ) -> None:
        self.state = state
        self.room_memory = room_memory
        self.tuning = tuning
        self.model = model
        self.interval = max(5.0, interval)
        self._objects = None
        self._force = False
        self._trigger_source = "timer"
        self._last_run = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit_objects(
        self, objects: dict[str, int], force: bool = False, trigger_source: str = "timer"
    ) -> None:
        with self._lock:
            self._objects = dict(objects)
            self._force = self._force or bool(force)
            if force:
                self._trigger_source = trigger_source
        if force:
            self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.interval)
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._lock:
                objects = self._objects
                force = self._force
                trigger_source = self._trigger_source if self._force else "timer"
                self._force = False
                self._trigger_source = "timer"
            if not objects:
                continue
            if force and trigger_source == "IMU":
                time.sleep(self.tuning.room_imu_delay_seconds)
            self.state.set_status(room_status="processing")
            try:
                room, confidence = self._classify(objects, trigger_source)
            except Exception as exc:
                self.state.set_status(room_status="error", room_error=str(exc))
                print(f"[room detection] {exc}", flush=True)
            else:
                self._last_run = time.time()
                if room:
                    self.state.set_status(
                        room=room,
                        room_confidence=confidence,
                        room_status="running",
                        room_updated_at=self._last_run,
                        room_error="",
                    )
                else:
                    self.state.set_status(
                        room_status="running",
                        room_updated_at=self._last_run,
                        room_error="",
                    )

    def _classify(self, objects: dict[str, int], trigger_source: str) -> tuple[str, float]:
        snapshot = self.state.snapshot()
        previous_room = str(snapshot.get("room") or "unknown")
        if self.room_memory.has_rooms():
            room, confidence, accepted = self.room_memory.recognise(
                trigger_source=trigger_source,
                previous_room=previous_room,
            )
            return (room if accepted else "", confidence)
        return self._qwen_classify(objects)

    def _qwen_classify(self, objects: dict[str, int]) -> tuple[str, float]:
        prompt = (
            f"You are a deterministic room classifier. Detected object counts: {json.dumps(objects)}. "
            "Choose the room best supported by the objects. Rules: bed, pillow, dresser or wardrobe strongly means bedroom; "
            "toilet, sink or bathtub means bathroom; refrigerator, oven, stove or microwave means kitchen; "
            "sofa or couch means living room; dining table means dining room; desk with computer means office; "
            "stairs means stairs. If several clues exist, choose the strongest matching room. "
            "Do not answer unknown when any strong clue exists. Return JSON only in exactly this form: "
            "{\"room\":\"bedroom\",\"confidence\":0.9}. "
            "room must be exactly one of bedroom, kitchen, bathroom, living room, dining room, hallway, office, garage, stairs, outdoor, unknown. "
            "Use unknown only when the object list is empty or contains no useful clue."
        )
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_ctx": 256},
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read())
        text = result.get("message", {}).get("content", "")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise RuntimeError(f"text model returned no JSON: {text[:160]}")
        data = json.loads(match.group(0))
        return (
            str(data.get("room", "unknown")).strip().lower(),
            max(0, min(1, float(data.get("confidence", 0)))),
        )
