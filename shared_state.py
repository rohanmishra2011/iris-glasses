"""Thread-safe snapshots shared by perception and the local dashboard."""
from __future__ import annotations

import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import cv2


class SharedState:
    """Hold only the newest dashboard data; never queue camera frames."""

    def __init__(self, jpeg_quality: int = 70, stream_fps: float = 8.0) -> None:
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._frame_version = 0
        self._last_encode = 0.0
        self._encode_interval = 1.0 / max(1.0, stream_fps)
        self._jpeg_quality = max(35, min(90, jpeg_quality))
        self._status: dict[str, Any] = {
            "camera": "starting",
            "yolo": "starting",
            "room": "unknown",
            "room_confidence": 0.0,
            "room_status": "starting",
            "room_updated_at": None,
            "room_error": "",
            "cpu_temperature_c": None,
            "supply_voltage_v": None,
            "power_status": "starting",
            "throttled_flags": None,
            "imu_status": "starting",
            "imu_address": None,
            "imu_accel_g": None,
            "imu_gyro_dps": None,
            "imu_temperature_c": None,
            "imu_updated_at": None,
            "imu_error": "",
            "voice_status": "starting",
            "last_command": "",
            "last_response": "",
            "last_command_at": None,
            "detections": [],
            "frame_id": 0,
            "last_update": None,
        }

    def set_status(self, **values: Any) -> None:
        with self._condition:
            self._status.update(values)
            self._condition.notify_all()

    def publish(self, frame, detections: list[dict[str, Any]], **values: Any) -> None:
        """Publish metadata every frame and an MJPEG frame at a capped rate."""
        now = time.monotonic()
        jpeg: bytes | None = None
        if now - self._last_encode >= self._encode_interval:
            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )
            if ok:
                jpeg = encoded.tobytes()

        with self._condition:
            self._status.update(values)
            self._status["detections"] = detections
            self._status["last_update"] = datetime.now(timezone.utc).isoformat()
            if jpeg is not None:
                self._jpeg = jpeg
                self._frame_version += 1
                self._last_encode = now
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return deepcopy(self._status)

    def wait_for_jpeg(
        self, previous_version: int, timeout: float = 2.0
    ) -> tuple[int, bytes | None]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._frame_version != previous_version, timeout=timeout
            )
            return self._frame_version, self._jpeg
