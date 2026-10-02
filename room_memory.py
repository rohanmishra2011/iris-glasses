"""Scanned-room fingerprints and landmark-based room recognition."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import threading
import time

from config import LANDMARKS, Tuning
from event_logger import EventLogger
from object_database import ALIASES, ObjectDatabase


class RoomMemory:
    def __init__(self, db_path: str | Path, tuning: Tuning, logger: EventLogger) -> None:
        self.db_path = Path(db_path)
        self.tuning = tuning
        self.logger = logger
        self.landmarks = tuple(LANDMARKS)
        self._history: deque[tuple[float, set[str]]] = deque()
        self._lock = threading.Lock()
        ObjectDatabase.ensure_schema(self.db_path)

    def observe(self, object_counts: dict[str, int], timestamp: float | None = None) -> None:
        now = time.monotonic() if timestamp is None else timestamp
        labels = {self._normalise(label) for label, count in object_counts.items() if count > 0}
        landmarks = {label for label in labels if label in self.landmarks}
        keep_after = now - max(self.tuning.room_scan_seconds, self.tuning.room_fingerprint_window_seconds) - 5.0
        with self._lock:
            self._history.append((now, landmarks))
            while self._history and self._history[0][0] < keep_after:
                self._history.popleft()

    def has_rooms(self) -> bool:
        with sqlite3.connect(self.db_path) as connection:
            ObjectDatabase._create_schema(connection)
            row = connection.execute("SELECT 1 FROM rooms LIMIT 1").fetchone()
        return row is not None

    def scan_room(self, room_name: str, start_time: float, end_time: float) -> dict[str, float]:
        fingerprint = self._fingerprint(start_time, end_time, self.tuning.room_landmark_min_fraction)
        clean_name = " ".join(room_name.strip().lower().split())
        if not clean_name or not fingerprint:
            return {}
        with sqlite3.connect(self.db_path) as connection:
            ObjectDatabase._create_schema(connection)
            connection.execute(
                """
                INSERT INTO rooms(name, fingerprint_json, updated_at)
                VALUES(?,?,?)
                ON CONFLICT(name) DO UPDATE SET
                    fingerprint_json=excluded.fingerprint_json,
                    updated_at=excluded.updated_at
                """,
                (
                    clean_name,
                    json.dumps(fingerprint, sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.commit()
        self.logger.log(
            "room_recognition",
            all_room_scores={clean_name: 1.0},
            chosen_room=clean_name,
            trigger_source="scan",
        )
        return fingerprint

    def recognise(self, trigger_source: str, previous_room: str = "unknown") -> tuple[str, float, bool]:
        saved = self._saved_rooms()
        if not saved:
            return previous_room, 0.0, False
        now = time.monotonic()
        current = self._fingerprint(
            now - self.tuning.room_fingerprint_window_seconds,
            now,
            minimum_fraction=0.0,
        )
        scores = {
            name: self._cosine_similarity(current, fingerprint)
            for name, fingerprint in saved.items()
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_name, best_score = ranked[0] if ranked else (previous_room, 0.0)
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        accepted = (
            best_score > self.tuning.room_threshold
            and best_score - second_score >= self.tuning.room_margin
        )
        chosen = best_name if accepted else previous_room
        self.logger.log(
            "room_recognition",
            all_room_scores=scores,
            chosen_room=chosen if accepted else "",
            trigger_source=trigger_source,
        )
        return chosen, best_score, accepted

    def _saved_rooms(self) -> dict[str, dict[str, float]]:
        with sqlite3.connect(self.db_path) as connection:
            ObjectDatabase._create_schema(connection)
            rows = connection.execute("SELECT name, fingerprint_json FROM rooms").fetchall()
        return {name: json.loads(fingerprint_json) for name, fingerprint_json in rows}

    def _fingerprint(
        self, start_time: float, end_time: float, minimum_fraction: float
    ) -> dict[str, float]:
        with self._lock:
            frames = [
                landmarks
                for timestamp, landmarks in self._history
                if start_time <= timestamp <= end_time
            ]
        if not frames:
            return {}
        counts = {label: 0 for label in self.landmarks}
        for landmarks in frames:
            for label in landmarks:
                if label in counts:
                    counts[label] += 1
        total = len(frames)
        return {
            label: count / total
            for label, count in counts.items()
            if count / total >= minimum_fraction
        }

    def _normalise(self, label: str) -> str:
        clean = " ".join(label.lower().strip().split())
        return ALIASES.get(clean, clean)

    @staticmethod
    def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
        keys = set(left) | set(right)
        if not keys:
            return 0.0
        dot = sum(left.get(key, 0.0) * right.get(key, 0.0) for key in keys)
        left_norm = math.sqrt(sum(left.get(key, 0.0) ** 2 for key in keys))
        right_norm = math.sqrt(sum(right.get(key, 0.0) ** 2 for key in keys))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)
