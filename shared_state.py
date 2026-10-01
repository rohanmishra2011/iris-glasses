"""Thread-safe runtime state shared by perception, IMU, room and voice workers."""
from __future__ import annotations

import threading
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


class SharedState:
    """Hold the newest runtime state without queueing camera frames."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
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
            "imu_mount": "waist",
            "imu_accel_magnitude_g": None,
            "imu_linear_accel_g": None,
            "imu_motion_state": "starting",
            "imu_step_count": 0,
            "imu_step_rate_hz": 0.0,
            "imu_yaw_rate_dps": 0.0,
            "imu_heading_delta_deg": 0.0,
            "imu_turn_angle_deg": 0.0,
            "imu_turn_progress_deg": 0.0,
            "imu_turn_direction": None,
            "imu_turn_event_direction": None,
            "imu_turn_event_id": 0,
            "imu_room_transition_likely": False,
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
        """Publish metadata for the latest frame."""
        with self._condition:
            self._status.update(values)
            self._status["detections"] = detections
            self._status["last_update"] = datetime.now(timezone.utc).isoformat()
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return deepcopy(self._status)
