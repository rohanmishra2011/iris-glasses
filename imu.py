"""Non-blocking MPU6050 reader and waist-mounted room-transition trigger."""
from __future__ import annotations

from collections import deque
import math
import threading
import time

from smbus2 import SMBus

from config import Tuning
from event_logger import EventLogger


class MPU6050Worker:
    ADDRESSES = (0x68, 0x69)
    SUPPORTED_IDENTITIES = (0x68, 0x70)

    def __init__(
        self,
        state,
        tuning: Tuning,
        logger: EventLogger,
        bus_number: int = 1,
    ) -> None:
        self.state = state
        self.tuning = tuning
        self.logger = logger
        self.bus_number = bus_number
        self.period = 1.0 / tuning.imu_sample_hz
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_sample_at: float | None = None
        self._gyro_z_bias = 0.0
        self._gyro_bias_count = 0
        self._gyro_bias_target = max(1, int(tuning.imu_calibration_seconds * tuning.imu_sample_hz))
        self._last_linear_accel_g = 0.0
        self._last_step_at = 0.0
        self._total_steps = 0
        self._transition_steps: deque[float] = deque()
        self._transition_yaw_deg = 0.0
        self._heading_delta_deg = 0.0
        self._last_transition_at = 0.0
        self._transition_event_id = 0
        self._blocked_triggers = 0

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
        linear_accel_g = abs(accel_magnitude_g - 1.0)
        if self._last_sample_at is None:
            dt = self.period
        else:
            dt = max(0.001, min(0.25, now - self._last_sample_at))
        self._last_sample_at = now

        if self._gyro_bias_count < self._gyro_bias_target:
            self._gyro_bias_count += 1
            self._gyro_z_bias += (gz - self._gyro_z_bias) / self._gyro_bias_count
            motion_state = "calibrating"
            yaw_rate_dps = 0.0
        else:
            yaw_rate_dps = gz - self._gyro_z_bias
            motion_state = "walking" if self._step_detected(linear_accel_g, now) else "steady"
            if abs(yaw_rate_dps) > self.tuning.imu_turn_rate_threshold_dps:
                self._transition_yaw_deg += yaw_rate_dps * dt
                self._heading_delta_deg += yaw_rate_dps * dt
            self._prune_transition_window(now)
            self._check_transition(now)

        step_rate_hz = len(self._transition_steps) / self.tuning.imu_transition_window_seconds
        return {
            "imu_mount": "waist",
            "imu_accel_magnitude_g": round(accel_magnitude_g, 3),
            "imu_linear_accel_g": round(linear_accel_g, 3),
            "imu_motion_state": motion_state,
            "imu_step_count": self._total_steps,
            "imu_step_rate_hz": round(step_rate_hz, 2),
            "imu_yaw_rate_dps": round(yaw_rate_dps, 2),
            "imu_heading_delta_deg": round(self._heading_delta_deg, 1),
            "imu_turn_angle_deg": round(self._transition_yaw_deg, 1),
            "imu_turn_progress_deg": round(self._transition_yaw_deg, 1),
            "imu_turn_direction": "left" if self._transition_yaw_deg > 0 else "right",
            "imu_turn_event_direction": None,
            "imu_turn_event_id": self._transition_event_id,
            "imu_room_transition_likely": now - self._last_transition_at <= 1.0,
        }

    def _step_detected(self, linear_accel_g: float, now: float) -> bool:
        rising_peak = (
            self._last_linear_accel_g <= self.tuning.imu_step_threshold_g
            and linear_accel_g > self.tuning.imu_step_threshold_g
        )
        enough_time = (
            now - self._last_step_at >= self.tuning.imu_step_refractory_seconds
        )
        self._last_linear_accel_g = linear_accel_g
        if not (rising_peak and enough_time):
            return False
        self._last_step_at = now
        self._total_steps += 1
        self._transition_steps.append(now)
        self.logger.log(
            "imu_events",
            event="step",
            steps=len(self._transition_steps),
            yaw=self._transition_yaw_deg,
            triggers=0,
            blocked_triggers=self._blocked_triggers,
        )
        return True

    def _prune_transition_window(self, now: float) -> None:
        cutoff = now - self.tuning.imu_transition_window_seconds
        while self._transition_steps and self._transition_steps[0] < cutoff:
            self._transition_steps.popleft()
        if not self._transition_steps:
            self._transition_yaw_deg = 0.0

    def _check_transition(self, now: float) -> None:
        enough_steps = len(self._transition_steps) >= self.tuning.imu_transition_steps
        enough_yaw = abs(self._transition_yaw_deg) >= self.tuning.imu_transition_yaw_deg
        if not (enough_steps and enough_yaw):
            return
        if now - self._last_transition_at < self.tuning.imu_transition_cooldown_seconds:
            self._blocked_triggers += 1
            self.logger.log(
                "imu_events",
                event="blocked_trigger",
                steps=len(self._transition_steps),
                yaw=self._transition_yaw_deg,
                triggers=self._transition_event_id,
                blocked_triggers=self._blocked_triggers,
            )
            self._reset_transition_window()
            return
        self._transition_event_id += 1
        self._last_transition_at = now
        self.logger.log(
            "imu_events",
            event="trigger",
            steps=len(self._transition_steps),
            yaw=self._transition_yaw_deg,
            triggers=self._transition_event_id,
            blocked_triggers=self._blocked_triggers,
        )
        self._reset_transition_window()

    def _reset_transition_window(self) -> None:
        self._transition_steps.clear()
        self._transition_yaw_deg = 0.0

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
                    self.state.set_status(
                        imu_status="calibrating",
                        imu_address=f"0x{address:02x}",
                    )
                    print("Calibrating, please stay still", flush=True)
                    self.logger.log(
                        "imu_events",
                        event="calibration_start",
                        steps=0,
                        yaw=0.0,
                        triggers=self._transition_event_id,
                        blocked_triggers=self._blocked_triggers,
                    )
                    while not self._stop.is_set():
                        data = bus.read_i2c_block_data(address, 0x3B, 14)
                        ax, ay, az = (
                            self._word(data, offset) / 16384.0 for offset in (0, 2, 4)
                        )
                        temperature = self._word(data, 6) / 340.0 + 36.53
                        gx, gy, gz = (
                            self._word(data, offset) / 131.0 for offset in (8, 10, 12)
                        )
                        now = time.time()
                        motion = self._estimate_motion(ax, ay, az, gz, now)
                        self.state.set_status(
                            imu_status=(
                                "calibrating"
                                if self._gyro_bias_count < self._gyro_bias_target
                                else "running"
                            ),
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
