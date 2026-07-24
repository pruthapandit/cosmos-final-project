"""IMU sensor reader.

Wraps the serial line format already used by python_imu_segment_demo_student.py
behind the BaseSensorReader interface, so it can run side by side with other
sensors under one coordinator clock in collect.py.

Expected serial line format (from the ESP32 Core2 IMU firmware):
    accel[g] x=<f> y=<f> z=<f> | gyro[dps] x=<f> y=<f> z=<f>
"""

from __future__ import annotations

import re
import time
from typing import Any

from sensors.base_reader import BaseSensorReader

FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
SAMPLE_RE = re.compile(
    rf"accel\[g\]\s+x=\s*({FLOAT_RE})\s+y=\s*({FLOAT_RE})\s+z=\s*({FLOAT_RE})"
    rf"\s+\|\s+gyro\[dps\]\s+x=\s*({FLOAT_RE})\s+y=\s*({FLOAT_RE})\s+z=\s*({FLOAT_RE})"
)


def parse_imu_line(line: str) -> dict[str, float] | None:
    """Parse one serial line into an IMU sample dict, or None if it doesn't match."""
    match = SAMPLE_RE.search(line)
    if not match:
        return None
    ax, ay, az, gx, gy, gz = (float(value) for value in match.groups())
    return {"ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz}


class ImuReader(BaseSensorReader):
    """Reads accel/gyro samples from the serial-connected IMU (ESP32 Core2)."""

    name = "imu"

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        print_unparsed: bool = False,
        max_rate_hz: float | None = 25.0,
    ) -> None:
        super().__init__()
        self.port = port
        self.baud = baud
        self.print_unparsed = print_unparsed
        # The board itself streams at whatever rate its firmware is built with
        # (~100Hz); this throttles what gets pushed downstream by dropping
        # samples that arrive before the minimum interval has elapsed, rather
        # than reflashing the firmware.
        self.min_sample_interval = (1.0 / max_rate_hz) if max_rate_hz else 0.0
        self._serial_port: Any | None = None
        self._last_pushed_at: float | None = None

    def _open(self) -> None:
        import serial  # local import: only required when this reader is actually used

        self._serial_port = serial.Serial(self.port, baudrate=self.baud, timeout=0.1)
        self._serial_port.reset_input_buffer()

    def _close(self) -> None:
        if self._serial_port is not None:
            self._serial_port.close()
            self._serial_port = None

    def _run(self) -> None:
        assert self._serial_port is not None
        while not self._stop_event.is_set():
            try:
                raw = self._serial_port.readline()
            except Exception as exc:  # serial disconnect, permission error, etc.
                self._push_error(f"imu read failed: {exc}")
                break

            if not raw:
                continue

            line = raw.decode("utf-8", errors="replace").strip()
            sample = parse_imu_line(line)
            if sample is None:
                if self.print_unparsed:
                    print(f"[imu] unparsed: {line}")
                continue

            now = time.monotonic()
            if self.min_sample_interval and self._last_pushed_at is not None:
                if now - self._last_pushed_at < self.min_sample_interval:
                    continue
            self._last_pushed_at = now
            self._push_sample(sample)


if __name__ == "__main__":
    # Quick manual smoke test: python3 -m sensors.imu_reader --port /dev/tty.xxxx
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Smoke-test the IMU reader alone.")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/cu.usbserial-5B1F0080901")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=5.0, help="How long to stream before stopping")
    args = parser.parse_args()

    reader = ImuReader(port=args.port, baud=args.baud, print_unparsed=True)
    reader.start()
    print(f"Streaming for {args.seconds:g}s... (Ctrl+C to stop early)")
    start = time.monotonic()
    total = 0
    try:
        while time.monotonic() - start < args.seconds:
            samples, errors = reader.drain()
            for err in errors:
                print(f"[imu] ERROR: {err}")
            for ts, fields in samples:
                total += 1
                if total % 10 == 0:
                    print(f"[imu] t={ts:.3f} {fields}")
            time.sleep(0.05)
    finally:
        reader.stop()
    print(f"Done. Received {total} samples in {args.seconds:g}s "
          f"(~{total / args.seconds:.1f} Hz).")
