"""Small runtime configuration loader for PUPIL testing thresholds."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Tuning:
    test_mode: bool = False

    # Personal object matching.
    match_threshold: float = 0.60
    object_match_min_interval_seconds: float = 1.0
    object_scan_crop_count: int = 5
    object_scan_seconds: float = 2.0
    personal_crop_size: int = 128
    hsv_h_bins: int = 30
    hsv_s_bins: int = 32
    orb_features: int = 500
    orb_ratio: float = 0.75
    orb_min_keypoints: int = 10

    # Room scanning and recognition.
    room_scan_seconds: float = 15.0
    room_fingerprint_window_seconds: float = 10.0
    room_landmark_min_fraction: float = 0.20
    room_threshold: float = 0.50
    room_margin: float = 0.10
    room_timer_seconds: float = 30.0
    room_imu_delay_seconds: float = 5.0

    # IMU transition trigger.
    imu_sample_hz: float = 20.0
    imu_calibration_seconds: float = 2.0
    imu_step_threshold_g: float = 0.16
    imu_step_refractory_seconds: float = 0.30
    imu_turn_rate_threshold_dps: float = 35.0
    imu_transition_steps: int = 4
    imu_transition_yaw_deg: float = 55.0
    imu_transition_window_seconds: float = 10.0
    imu_transition_cooldown_seconds: float = 5.0


LANDMARKS = (
    "bed",
    "couch",
    "chair",
    "dining table",
    "refrigerator",
    "oven",
    "microwave",
    "sink",
    "toilet",
    "tv",
    "potted plant",
    "bench",
    "clock",
    "watch",
)


def load_tuning(path: str | Path = "configs/default.toml") -> Tuning:
    config_path = Path(path)
    if not config_path.exists():
        return Tuning()
    with config_path.open("rb") as config_file:
        data = tomllib.load(config_file)
    values = data.get("tuning", {})
    allowed = {field.name for field in fields(Tuning)}
    clean = {key: value for key, value in values.items() if key in allowed}
    return Tuning(**clean)
