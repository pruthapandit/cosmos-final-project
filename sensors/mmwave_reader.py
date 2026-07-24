"""mmWave sensor reader.

Wraps the CLI handshake and TLV/frame parsing that already work in
get_range_profile.py behind the BaseSensorReader interface, instead of
reimplementing any of that logic.

Place this file's parent 'sensors/' folder directly inside your existing
mmWave project directory (the one that contains get_range_profile.py), so
the import below resolves.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sensors.base_reader import BaseSensorReader

# Reuses your existing, working serial/TLV code -- nothing below reimplements
# the handshake or frame parsing.
import get_range_profile as grp


class MmwaveReader(BaseSensorReader):
    """Streams range-profile frames from the radar via the existing get_range_profile.py code."""

    name = "mmwave"

    def __init__(
        self,
        port: str,
        cfg_path: Path,
        baud: int = 115200,
        frame_timeout: float = 5.0,
        use_cfg_baud_rate: bool = False,
    ) -> None:
        super().__init__()
        self.port_name = port
        self.cfg_path = Path(cfg_path)
        self.baud = baud
        self.frame_timeout = frame_timeout
        self.use_cfg_baud_rate = use_cfg_baud_rate

        self._serial_port: Any | None = None
        self._commands: list[str] | None = None
        self.range_config: Any | None = None
        self._expected_range_profile_bytes: int | None = None

    def _open(self) -> None:
        import serial

        self._commands = grp.load_configuration(self.cfg_path)
        self.range_config = grp.parse_range_config(self._commands)
        self._expected_range_profile_bytes = (
            None if self.range_config is None else self.range_config.num_range_bins * 4
        )

        self._serial_port = serial.Serial(self.port_name, baudrate=self.baud, timeout=0.2)
        time.sleep(0.5)
        grp.send_configuration(self._serial_port, self._commands, self.use_cfg_baud_rate)

    def _close(self) -> None:
        if self._serial_port is not None:
            try:
                grp.write_cli_command(self._serial_port, "sensorStop 0")
                time.sleep(0.2)
            except Exception:
                pass  # best-effort stop -- don't block shutdown on a CLI hiccup
            self._serial_port.close()
            self._serial_port = None

    def _run(self) -> None:
        assert self._serial_port is not None
        consecutive_warnings = 0
        while not self._stop_event.is_set():
            try:
                frame_number, tlvs = grp.read_frame(
                    self._serial_port,
                    self.frame_timeout,
                    self._expected_range_profile_bytes,
                )
            except (TimeoutError, ValueError, RuntimeError) as exc:
                consecutive_warnings += 1
                self._push_error(f"mmwave frame warning: {exc}")
                if consecutive_warnings >= 5:
                    self._push_error("mmwave: too many consecutive frame warnings, stopping reader")
                    return
                continue

            consecutive_warnings = 0

            for tlv_type, payload in tlvs:
                if tlv_type not in {grp.RANGE_PROFILE_MAJOR, grp.RANGE_PROFILE_MINOR}:
                    continue
                if len(payload) % 4 != 0:
                    continue

                import numpy as np

                profile = np.frombuffer(payload, dtype="<u4").astype(float)
                # Stored as a JSON string so it fits one CSV column like every
                # other sensor's fields; unpack with json.loads() downstream
                # during feature extraction.
                self._push_sample({
                    "frame_number": frame_number,
                    "profile_json": json.dumps(profile.tolist()),
                })


if __name__ == "__main__":
    # Quick manual smoke test:
    #   python3 -m sensors.mmwave_reader --port /dev/tty.xxxx --cfg path/to/config.cfg
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the mmWave reader alone.")
    parser.add_argument("--port", required=True)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--use-cfg-baud-rate", action="store_true")
    args = parser.parse_args()

    reader = MmwaveReader(
        port=args.port,
        cfg_path=args.cfg,
        baud=args.baud,
        use_cfg_baud_rate=args.use_cfg_baud_rate,
    )
    reader.start()
    print(f"Streaming for {args.seconds:g}s... (Ctrl+C to stop early)")
    start = time.monotonic()
    total = 0
    try:
        while time.monotonic() - start < args.seconds:
            samples, errors = reader.drain()
            for err in errors:
                print(f"[mmwave] {err}")
            total += len(samples)
            if samples:
                last_t, last_fields = samples[-1]
                print(f"[mmwave] t={last_t:.3f} frame={last_fields['frame_number']} "
                      f"profile_len={len(json.loads(last_fields['profile_json']))}")
            time.sleep(0.05)
    finally:
        reader.stop()
    print(f"Done. Received {total} frames in {args.seconds:g}s (~{total / args.seconds:.1f} fps).")
