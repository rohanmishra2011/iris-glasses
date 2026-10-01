"""Non-blocking MPU6050 reader and waist-motion estimator.

The IMU is mounted on the user's waist, so this module intentionally avoids
claiming exact head gaze or exact position. It only extracts coarse, useful
events from the accelerometer/gyroscope:

- stationary / moving / walking from gravity-compensated acceleration magnitude;
- step-like waist motion from acceleration peaks;
- left/right turn events from integrated yaw-rate;
- a "room transition likely" flag when a walking user makes a large turn.
"""
from __future__ import annotations

from collections import deque
import math
import threading
import time

from smbus2 import SMBus


class MPU6050Worker:
    ADDRESSES = (0x68, 0x69)
    # Genuine MPU6050 parts report 0x68. This GY-521-labelled board reports
    # 0x70 but exposes the same register layout and produces valid motion data.
    SUPPORTED_IDENTITIES = (0x68, 0x70)

    # Waist-mounted motion thresholds. These are intentionally conservative:
    # they detect useful events without pretending to do exact localization.
    GYRO_BIAS_SAMPLES = 40
    GYRO_DEADBAND_DPS = 2.5
    WALK_LINEAR_THRESHOLD_G = 0.10
    STATIONARY_LINEAR_THRESHOLD_G = 0.045
    STATIONARY_SECONDS = 1.4
    STEP_PEAK_G = 0.16
    STEP_COOLDOWN_SECONDS = 0.28
    WALK_WINDOW_SECONDS = 3.0
    WALK_MIN_STEPS_IN_WINDOW = 2
    TURN_RATE_THRESHOLD_DPS = 35.0
    TURN_END_RATE_DPS = 14.0
    TURN_MIN_ANGLE_DEG = 55.0
    TURN_COOLDOWN_SECONDS = 1.25
    ROOM_TRANSITION_FLAG_SECONDS = 2.0

    def __init__(self, state, bus_number: int = 1, sample_hz: float = 20.0) -> None:
        self.state = state
        self.bus_number = bus_number
        self.period = 1.0 / sample_hz
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_sample_at: float | None = None
        self._gyro_z_bias = 0.0
        self._gyro_bias_count = 0
        self._last_linear_accel_g = 0.0
        self._stationary_since: float | None = None
        self._step_times: deque[float] = deque()
        self._total_steps = 0
        self._last_step_at = 0.0
        self._turn_angle_deg = 0.0
        self._last_turn_at = 0.0
        self._turn_event_id = 0
        self._heading_delta_deg = 0.0
        self._last_turn_direction: str | None = None
        self._last_completed_turn_angle_deg = 0.0
        self._last_transition_at = 0.0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    @staticmethod
    def _signed(value: int) -> int:
        return value - 65536 if value & 0x8000 else value

    @classmethod
    def _word(cls, data: list[int], offset: int) -> int:
        return cls._signed((data[offset] << 8) | data[offset + 1])

    def _find(self, bus: SMBus) -> int | None:
        for address in self.ADDRESSES:
            try:
                if bus.read_byte_data(address, 0x75) in self.SUPPORTED_IDENTITIES:
                    return address
            except OSError:
                continue
        return None

    def _estimate_motion(
        self, ax: float, ay: float, az: float, gz: float, now: float
    ) -> dict[str, object]:
        accel_magnitude_g = math.sqrt(ax * ax + ay * ay + az * az)
        # Subtract the static 1 g gravity magnitude. This is orientation-tolerant
        # for a waist mount, unlike relying on one fixed accelerometer axis.
        linear_accel_g = abs(accel_magnitude_g - 1.0)

        if self._last_sample_at is None:
            dt = self.period
        else:
            dt = max(0.001, min(0.25, now - self._last_sample_at))
        self._last_sample_at = now

        if self._gyro_bias_count < self.GYRO_BIAS_SAMPLES:
            self._gyro_bias_count += 1
            self._gyro_z_bias += (gz - self._gyro_z_bias) / self._gyro_bias_count
            motion_state = "calibrating"
            yaw_rate_dps = 0.0
        else:
            yaw_rate_dps = gz - self._gyro_z_bias
            if abs(yaw_rate_dps) < self.GYRO_DEADBAND_DPS:
                yaw_rate_dps = 0.0
            motion_state = self._classify_motion(linear_accel_g, now)

        self._detect_step(linear_accel_g, now)
        self._heading_delta_deg += yaw_rate_dps * dt
        turn = self._detect_turn(yaw_rate_dps, dt, motion_state, now)

        while self._step_times and now - self._step_times[0] > self.WALK_WINDOW_SECONDS:
            self._step_times.popleft()
        step_rate_hz = len(self._step_times) / self.WALK_WINDOW_SECONDS

        if turn["event_direction"] is not None and motion_state in {"walking", "moving"}:
            self._last_transition_at = now
        room_transition_likely = now - self._last_transition_at <= self.ROOM_TRANSITION_FLAG_SECONDS

        return {
            "imu_mount": "waist",
            "imu_accel_magnitude_g": round(accel_magnitude_g, 3),
            "imu_linear_accel_g": round(linear_accel_g, 3),
            "imu_motion_state": motion_state,
            "imu_step_count": self._total_steps,
            "imu_step_rate_hz": round(step_rate_hz, 2),
            "imu_yaw_rate_dps": round(yaw_rate_dps, 2),
            "imu_heading_delta_deg": round(self._heading_delta_deg, 1),
            "imu_turn_angle_deg": round(turn["last_angle_deg"], 1),
            "imu_turn_progress_deg": round(turn["progress_angle_deg"], 1),
            "imu_turn_direction": turn["last_direction"],
            "imu_turn_event_direction": turn["event_direction"],
            "imu_turn_event_id": self._turn_event_id,
            "imu_room_transition_likely": room_transition_likely,
        }

    def _classify_motion(self, linear_accel_g: float, now: float) -> str:
        if linear_accel_g <= self.STATIONARY_LINEAR_THRESHOLD_G:
            if self._stationary_since is None:
                self._stationary_since = now
            if now - self._stationary_since >= self.STATIONARY_SECONDS:
                return "stationary"
        else:
            self._stationary_since = None

        recent_steps = sum(
            1 for step_time in self._step_times
            if now - step_time <= self.WALK_WINDOW_SECONDS
        )
        if recent_steps >= self.WALK_MIN_STEPS_IN_WINDOW:
            return "walking"
        if linear_accel_g >= self.WALK_LINEAR_THRESHOLD_G:
            return "moving"
        return "steady"

    def _detect_step(self, linear_accel_g: float, now: float) -> None:
        rising_peak = (
            self._last_linear_accel_g < self.STEP_PEAK_G
            and linear_accel_g >= self.STEP_PEAK_G
        )
        enough_time = now - self._last_step_at >= self.STEP_COOLDOWN_SECONDS
        if self._gyro_bias_count >= self.GYRO_BIAS_SAMPLES and rising_peak and enough_time:
            self._step_times.append(now)
            self._total_steps += 1
            self._last_step_at = now
        self._last_linear_accel_g = linear_accel_g

    def _detect_turn(
        self, yaw_rate_dps: float, dt: float, motion_state: str, now: float
    ) -> dict[str, object]:
        if abs(yaw_rate_dps) >= self.TURN_RATE_THRESHOLD_DPS:
            self._turn_angle_deg += yaw_rate_dps * dt
        elif abs(yaw_rate_dps) <= self.TURN_END_RATE_DPS:
            # If the user stopped rotating before a meaningful turn, slowly
            # bleed off the partial angle instead of creating false turns.
            if abs(self._turn_angle_deg) < self.TURN_MIN_ANGLE_DEG:
                self._turn_angle_deg *= 0.82

        completed_angle = self._turn_angle_deg
        event_direction = None
        if (
            abs(completed_angle) >= self.TURN_MIN_ANGLE_DEG
            and now - self._last_turn_at >= self.TURN_COOLDOWN_SECONDS
            and motion_state != "stationary"
        ):
            # Positive/negative depends on board orientation. If testing shows
            # left/right is flipped, swap these two labels; the math is still
            # correctly detecting a waist yaw event.
            event_direction = "left" if completed_angle > 0 else "right"
            self._last_turn_direction = event_direction
            self._last_completed_turn_angle_deg = completed_angle
            self._turn_event_id += 1
            self._last_turn_at = now
            self._turn_angle_deg = 0.0

        return {
            "event_direction": event_direction,
            "last_direction": self._last_turn_direction,
            "last_angle_deg": self._last_completed_turn_angle_deg,
            "progress_angle_deg": self._turn_angle_deg,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with SMBus(self.bus_number) as bus:
                    address = self._find(bus)
                    if address is None:
                        self.state.set_status(imu_status="disconnected", imu_address=None)
                        self._stop.wait(2.0)
                        continue
                    bus.write_byte_data(address, 0x6B, 0x00)
                    self.state.set_status(imu_status="running", imu_address=f"0x{address:02x}")
                    while not self._stop.is_set():
                        data = bus.read_i2c_block_data(address, 0x3B, 14)
                        ax, ay, az = (self._word(data, offset) / 16384.0 for offset in (0, 2, 4))
                        temperature = self._word(data, 6) / 340.0 + 36.53
                        gx, gy, gz = (self._word(data, offset) / 131.0 for offset in (8, 10, 12))
                        now = time.time()
                        motion = self._estimate_motion(ax, ay, az, gz, now)
                        self.state.set_status(
                            imu_status="running",
                            imu_address=f"0x{address:02x}",
                            imu_accel_g={"x": round(ax, 3), "y": round(ay, 3), "z": round(az, 3)},
                            imu_gyro_dps={"x": round(gx, 2), "y": round(gy, 2), "z": round(gz, 2)},
                            imu_temperature_c=round(temperature, 1),
                            imu_updated_at=now,
                            imu_error="",
                            **motion,
                        )
                        self._stop.wait(self.period)
            except (OSError, ValueError) as exc:
                self.state.set_status(imu_status="error", imu_error=str(exc))
                self._stop.wait(2.0)
