"""Non-blocking MPU6050 reader for Raspberry Pi I2C bus 1."""
from __future__ import annotations

import threading
import time

from smbus2 import SMBus


class MPU6050Worker:
    ADDRESSES = (0x68, 0x69)
    # Genuine MPU6050 parts report 0x68. This GY-521-labelled board reports
    # 0x70 but exposes the same register layout and produces valid motion data.
    SUPPORTED_IDENTITIES = (0x68, 0x70)

    def __init__(self, state, bus_number: int = 1, sample_hz: float = 20.0) -> None:
        self.state = state
        self.bus_number = bus_number
        self.period = 1.0 / sample_hz
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

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
                        self.state.set_status(
                            imu_status="running",
                            imu_address=f"0x{address:02x}",
                            imu_accel_g={"x": round(ax, 3), "y": round(ay, 3), "z": round(az, 3)},
                            imu_gyro_dps={"x": round(gx, 2), "y": round(gy, 2), "z": round(gz, 2)},
                            imu_temperature_c=round(temperature, 1),
                            imu_updated_at=time.time(),
                            imu_error="",
                        )
                        self._stop.wait(self.period)
            except (OSError, ValueError) as exc:
                self.state.set_status(imu_status="error", imu_error=str(exc))
                self._stop.wait(2.0)
