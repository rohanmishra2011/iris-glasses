"""Background Raspberry Pi health telemetry for the local dashboard."""
from __future__ import annotations

import re
import subprocess
import threading
from pathlib import Path


class SystemHealthWorker:
    def __init__(self, state, interval: float = 5.0) -> None:
        self.state = state
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    @staticmethod
    def _command(*args: str) -> str:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=2.0, check=True
        ).stdout

    def _sample(self) -> dict[str, object]:
        values: dict[str, object] = {
            "cpu_temperature_c": None,
            "supply_voltage_v": None,
            "power_status": "unavailable",
            "throttled_flags": None,
        }
        try:
            values["cpu_temperature_c"] = round(
                float(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000, 1
            )
        except (OSError, ValueError):
            pass
        try:
            adc = self._command("vcgencmd", "pmic_read_adc")
            match = re.search(r"EXT5V_V\s+volt\(\d+\)=([0-9.]+)V", adc)
            if match:
                values["supply_voltage_v"] = round(float(match.group(1)), 3)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        try:
            raw = self._command("vcgencmd", "get_throttled").strip()
            flags = int(raw.split("=", 1)[1], 16)
            values["throttled_flags"] = f"0x{flags:x}"
            if flags & 0x1:
                values["power_status"] = "undervoltage_now"
            elif flags & 0x10000:
                values["power_status"] = "undervoltage_occurred"
            elif flags:
                values["power_status"] = "throttling_detected"
            else:
                values["power_status"] = "healthy"
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
        return values

    def _run(self) -> None:
        while not self._stop.is_set():
            self.state.set_status(**self._sample())
            self._stop.wait(self.interval)
