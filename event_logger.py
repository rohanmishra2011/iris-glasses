"""CSV event logging for testing PUPIL offline."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any


class EventLogger:
    FILES = {
        "object_matches": (
            "timestamp",
            "yolo_class",
            "personal_object",
            "colour_score",
            "orb_score",
            "combined_score",
            "decision",
        ),
        "room_recognition": (
            "timestamp",
            "all_room_scores",
            "chosen_room",
            "trigger_source",
        ),
        "imu_events": (
            "timestamp",
            "event",
            "steps",
            "yaw",
            "triggers",
            "blocked_triggers",
        ),
        "queries": (
            "timestamp",
            "question",
            "answer",
            "latency_seconds",
        ),
    }

    def __init__(self, directory: str | Path = "logs", test_mode: bool = False) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.test_mode = test_mode
        self._lock = threading.Lock()
        self._ready: set[str] = set()

    def log(self, name: str, **row: Any) -> None:
        if name not in self.FILES:
            raise ValueError(f"unknown log file: {name}")
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), **row}
        fields = self.FILES[name]
        path = self.directory / f"{name}.csv"
        with self._lock:
            write_header = name not in self._ready and (
                not path.exists() or path.stat().st_size == 0
            )
            with path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                if write_header:
                    writer.writeheader()
                writer.writerow({field: self._format(row.get(field, "")) for field in fields})
            self._ready.add(name)
        if self.test_mode:
            printable = ", ".join(f"{key}={self._format(value)}" for key, value in row.items())
            print(f"[{name}] {printable}", flush=True)

    @staticmethod
    def _format(value: Any) -> str:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, sort_keys=True)
        if isinstance(value, float):
            return f"{value:.4f}"
        return "" if value is None else str(value)
