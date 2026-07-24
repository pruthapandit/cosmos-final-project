#!/usr/bin/env python3
"""Coordinator-based multi-sensor data collector.

Implements the architecture from the instructor's "Final Project
Implementation Hints" deck:
  - one coordinator process owns the clock (time.monotonic())
  - the coordinator writes event markers: trial_start, trial_end,
    trial_accept, trial_reject
  - each sensor reader timestamps its own samples against that same clock
  - everything is cut into trials offline using the event timestamps

Session layout written to --out (default ./data/raw/):
    session_YYYYMMDD_HHMMSS/
        session_metadata.json
        events.csv
        imu.csv
        mmwave.csv
        uwb.csv      (right_distance_cm/left_distance_cm per ranging round)
        trials.csv

Wired sensors: imu, mmwave, uwb. Run with any combination via --sensors,
e.g. --sensors imu mmwave uwb, against one shared clock.

This file (and the sensors/ package next to it) is meant to live inside
your existing mmWave project directory, alongside get_range_profile.py --
mmwave_reader.py imports that module directly rather than duplicating its
serial handshake or TLV parsing.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from sensors.imu_reader import ImuReader
from sensors.mmwave_reader import MmwaveReader
from sensors.uwb_reader import UwbReader

GESTURES = [
    "Pull", "Push", "Clockwise", "Anti-clockwise", "Right", "Left",
    "Bye-Bye", "One-Arm Boxing", "Clapping", "Two-Arm Boxing", "T-Arm",
    "Raise Arms", "Soli", "Making Fist and Open", "Palm Up-Down",
]  # full official set; pass a smaller subset via --gestures to start
   # (e.g. --gestures Pull Push Right Left), then expand once the
   # pipeline is proven


class SessionWriter:
    """Owns the session folder and the shared CSV files for one collection run."""

    def __init__(self, out_root: Path, sensor_names: list[str], gestures: list[str], collector: str) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = out_root / f"session_{stamp}"
        self.session_dir.mkdir(parents=True, exist_ok=False)

        self.events_path = self.session_dir / "events.csv"
        self.trials_path = self.session_dir / "trials.csv"
        self.sensor_paths = {name: self.session_dir / f"{name}.csv" for name in sensor_names}
        self._sensor_files: dict[str, Any] = {}
        self._sensor_writers: dict[str, csv.writer] = {}
        self._sensor_headers_written: dict[str, bool] = {name: False for name in sensor_names}

        self.events_file = open(self.events_path, "w", newline="")
        self.events_writer = csv.writer(self.events_file)
        self.events_writer.writerow(["t_monotonic", "event", "trial_index", "gesture", "note"])

        self.trials_file = open(self.trials_path, "w", newline="")
        self.trials_writer = csv.writer(self.trials_file)
        self.trials_writer.writerow(["trial_index", "gesture", "t_start", "t_end", "accepted"])

        with open(self.session_dir / "session_metadata.json", "w") as f:
            json.dump(
                {
                    "created_at": stamp,
                    "collector": collector,
                    "sensors": sensor_names,
                    "gestures": gestures,
                },
                f,
                indent=2,
            )

    def log_event(self, event: str, trial_index: int | None = None, gesture: str | None = None, note: str = "") -> float:
        t = time.monotonic()
        self.events_writer.writerow([f"{t:.6f}", event, trial_index if trial_index is not None else "", gesture or "", note])
        self.events_file.flush()
        return t

    def log_trial(self, trial_index: int, gesture: str, t_start: float, t_end: float, accepted: bool) -> None:
        self.trials_writer.writerow([trial_index, gesture, f"{t_start:.6f}", f"{t_end:.6f}", int(accepted)])
        self.trials_file.flush()

    def log_samples(self, sensor_name: str, samples: list[tuple[float, dict[str, Any]]]) -> None:
        if not samples:
            return
        if sensor_name not in self._sensor_files:
            self._sensor_files[sensor_name] = open(self.sensor_paths[sensor_name], "w", newline="")
            self._sensor_writers[sensor_name] = csv.writer(self._sensor_files[sensor_name])

        writer = self._sensor_writers[sensor_name]
        if not self._sensor_headers_written[sensor_name]:
            header = ["t_monotonic"] + sorted(samples[0][1].keys())
            writer.writerow(header)
            self._sensor_headers_written[sensor_name] = True

        for t, fields in samples:
            header = ["t_monotonic"] + sorted(fields.keys())
            row = [f"{t:.6f}"] + [fields[key] for key in header[1:]]
            writer.writerow(row)
        self._sensor_files[sensor_name].flush()

    def close(self) -> None:
        self.events_file.close()
        self.trials_file.close()
        for f in self._sensor_files.values():
            f.close()


def build_readers(sensor_names: list[str], args: argparse.Namespace) -> dict[str, Any]:
    readers: dict[str, Any] = {}
    if "imu" in sensor_names:
        readers["imu"] = ImuReader(port=args.imu_port, baud=args.imu_baud)
    if "mmwave" in sensor_names:
        readers["mmwave"] = MmwaveReader(
            port=args.mmwave_port,
            cfg_path=args.mmwave_cfg,
            baud=args.mmwave_baud,
            frame_timeout=args.mmwave_frame_timeout,
            use_cfg_baud_rate=args.mmwave_use_cfg_baud_rate,
        )
    if "uwb" in sensor_names:
        readers["uwb"] = UwbReader(
            controller_port=args.uwb_controller_port,
            right_port=args.uwb_right_port,
            left_port=args.uwb_left_port,
            uwb_tools_root=args.uwb_tools_root,
            channel=args.uwb_channel,
            preamble_code=args.uwb_preamble_idx,
            session_id=args.uwb_session,
            slot_span=args.uwb_slot_span,
            slots_per_rr=args.uwb_slots_per_rr,
            ranging_span=args.uwb_ranging_span,
            startup_delay=args.uwb_startup_delay,
        )
    return readers


def run_trial(
    readers: dict[str, Any],
    writer: SessionWriter,
    trial_index: int,
    gesture: str,
    trial_seconds: float,
) -> None:
    print(f"\nTrial {trial_index}: {gesture} -- press Enter to start ({trial_seconds:g}s recording)...")
    input()

    t_start = writer.log_event("trial_start", trial_index, gesture)
    end_at = t_start + trial_seconds
    while time.monotonic() < end_at:
        remaining = end_at - time.monotonic()
        print(f"  recording... {remaining:4.1f}s left", end="\r")
        time.sleep(0.1)
    t_end = writer.log_event("trial_end", trial_index, gesture)
    print(f"\n  captured {t_end - t_start:.2f}s")

    for name, reader in readers.items():
        samples, errors = reader.drain()
        for err in errors:
            print(f"  [{name}] ERROR: {err}")
        writer.log_samples(name, samples)
        print(f"  [{name}] {len(samples)} samples")

    decision = input("  keep this trial? [Y/n/r=repeat] ").strip().lower()
    if decision == "r":
        writer.log_event("trial_reject", trial_index, gesture, note="repeat requested")
        writer.log_trial(trial_index, gesture, t_start, t_end, accepted=False)
        run_trial(readers, writer, trial_index, gesture, trial_seconds)
        return

    accepted = decision != "n"
    writer.log_event("trial_accept" if accepted else "trial_reject", trial_index, gesture)
    writer.log_trial(trial_index, gesture, t_start, t_end, accepted=accepted)


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinator-based multi-sensor gesture data collector.")
    parser.add_argument("--sensors", nargs="+", default=["imu"], choices=["imu", "mmwave", "uwb"],
                         help="Which sensors to run this session.")
    parser.add_argument("--imu-port", default=None, help="Serial port for the IMU, e.g. /dev/cu.usbserial-XXXX")
    parser.add_argument("--imu-baud", type=int, default=115200)
    parser.add_argument("--mmwave-port", default=None, help="Serial port for the radar")
    parser.add_argument("--mmwave-cfg", type=Path, default=None, help="Radar CLI config file (.cfg)")
    parser.add_argument("--mmwave-baud", type=int, default=115200)
    parser.add_argument("--mmwave-frame-timeout", type=float, default=5.0)
    parser.add_argument("--mmwave-use-cfg-baud-rate", action="store_true")
    parser.add_argument("--uwb-controller-port", default=None, help="Serial port for the UWB anchor/controller")
    parser.add_argument("--uwb-right-port", default=None, help="Serial port for the right-wrist UWB tag (mac=1)")
    parser.add_argument("--uwb-left-port", default=None, help="Serial port for the left-wrist UWB tag (mac=2)")
    parser.add_argument(
        "--uwb-tools-root",
        type=Path,
        default=Path(r"C:\Users\Prutha Pandit\UWB_lab\uwb-qorvo-tools"),
        help="Path to the uwb-qorvo-tools folder (run_fira_twr.py lives under scripts/fira/run_fira_twr/).",
    )
    parser.add_argument("--uwb-channel", type=int, default=5, help="Class-sheet-assigned FiRa UWB channel.")
    parser.add_argument("--uwb-preamble-idx", type=int, default=12, help="Class-sheet-assigned FiRa preamble code.")
    parser.add_argument("--uwb-session", type=int, default=42)
    parser.add_argument("--uwb-slot-span", type=int, default=2400)
    parser.add_argument("--uwb-slots-per-rr", type=int, default=25)
    parser.add_argument("--uwb-ranging-span", type=int, default=200, help="Ranging interval in ms (FPS = 1000/this).")
    parser.add_argument("--uwb-startup-delay", type=float, default=5.0,
                         help="Seconds to wait after starting the two controlees before starting the controller.")
    parser.add_argument("--gestures", nargs="+", default=GESTURES)
    parser.add_argument("--trials-per-gesture", type=int, default=8)
    parser.add_argument("--trial-seconds", type=float, default=2.0)
    parser.add_argument("--collector", required=True, help="Name of the person performing gestures this session")
    parser.add_argument("--out", default="data/raw", help="Root output directory")
    args = parser.parse_args()

    if "imu" in args.sensors and not args.imu_port:
        parser.error("--imu-port is required when collecting from 'imu'")
    if "mmwave" in args.sensors and (not args.mmwave_port or not args.mmwave_cfg):
        parser.error("--mmwave-port and --mmwave-cfg are required when collecting from 'mmwave'")
    if "uwb" in args.sensors and not (args.uwb_controller_port and args.uwb_right_port and args.uwb_left_port):
        parser.error("--uwb-controller-port, --uwb-right-port, and --uwb-left-port are required when collecting from 'uwb'")

    readers = build_readers(args.sensors, args)
    writer = SessionWriter(Path(args.out), list(readers.keys()), args.gestures, args.collector)
    print(f"Session folder: {writer.session_dir}")

    for reader in readers.values():
        reader.start()

    try:
        trial_index = 0
        for gesture in args.gestures:
            for _ in range(args.trials_per_gesture):
                trial_index += 1
                run_trial(readers, writer, trial_index, gesture, args.trial_seconds)
    finally:
        for reader in readers.values():
            reader.stop()
        writer.close()

    print(f"\nDone. Session saved to {writer.session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
