#!/usr/bin/env python3
"""Cut per-trial gesture data out of collect.py sessions into one dataset.

collect.py records continuous imu.csv / mmwave.csv streams per session, plus
a trials.csv of trial_index/gesture/t_start/t_end/accepted derived from the
event markers. This script slices each accepted trial's rows out of those
continuous streams by t_monotonic, saves one .npz per trial, and writes a
combined manifest across every session (and therefore every collector), so a
later leave-one-person-out split stays possible just by filtering on the
"collector" column.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

FIELDNAMES = [
    "dataset_name",
    "source_session",
    "collector",
    "gesture",
    "trial_index",
    "t_start",
    "t_end",
    "duration_s",
    "imu_sample_count",
    "imu_mean_rate_hz",
    "mmwave_frame_count",
    "mmwave_mean_rate_hz",
    "uwb_round_count",
    "uwb_mean_rate_hz",
    "npz_path",
    "session_dir",
]


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return label.strip("_") or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut accepted trials out of collect.py sessions and combine into one gesture dataset."
    )
    parser.add_argument(
        "sessions",
        nargs="+",
        help="Session folders created by collect.py (data/raw/session_*).",
    )
    parser.add_argument(
        "--output",
        default=str(Path("datasets") / f"gesture_dataset_{timestamp()}"),
        help="Output dataset folder. Default: datasets/gesture_dataset_<timestamp>",
    )
    parser.add_argument(
        "--collector",
        action="append",
        help="Keep only this collector. Can be repeated or comma separated.",
    )
    parser.add_argument(
        "--gesture",
        action="append",
        help="Keep only this gesture. Can be repeated or comma separated.",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Also include trials the collector marked as rejected.",
    )
    return parser.parse_args()


def normalize_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    normalized: set[str] = set()
    for value in values:
        for part in value.split(","):
            label = part.strip()
            if label:
                normalized.add(label)
    return normalized


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def slice_imu(rows: list[dict[str, str]], t_start: float, t_end: float) -> dict[str, np.ndarray]:
    time_s: list[float] = []
    columns: dict[str, list[float]] = {name: [] for name in ("ax", "ay", "az", "gx", "gy", "gz")}

    for row in rows:
        t = float(row["t_monotonic"])
        if t < t_start or t > t_end:
            continue
        time_s.append(t - t_start)
        for name in columns:
            columns[name].append(float(row[name]))

    result = {"time_s": np.array(time_s, dtype=float)}
    result.update({name: np.array(values, dtype=float) for name, values in columns.items()})
    return result


def slice_mmwave(rows: list[dict[str, str]], t_start: float, t_end: float) -> dict[str, np.ndarray]:
    time_s: list[float] = []
    frame_number: list[int] = []
    profiles: list[list[float]] = []

    for row in rows:
        t = float(row["t_monotonic"])
        if t < t_start or t > t_end:
            continue
        time_s.append(t - t_start)
        frame_number.append(int(row["frame_number"]))
        profiles.append(json.loads(row["profile_json"]))

    if not profiles:
        range_profile = np.empty((0, 0), dtype=float)
    else:
        bin_count = max(len(profile) for profile in profiles)
        range_profile = np.zeros((len(profiles), bin_count), dtype=float)
        for index, profile in enumerate(profiles):
            range_profile[index, : len(profile)] = profile

    return {
        "time_s": np.array(time_s, dtype=float),
        "frame_number": np.array(frame_number, dtype=np.uint32),
        "range_profile": range_profile,
    }


def _float_or_nan(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def slice_uwb(rows: list[dict[str, str]], t_start: float, t_end: float) -> dict[str, np.ndarray]:
    time_s: list[float] = []
    sequence: list[int] = []
    right_distance_cm: list[float] = []
    left_distance_cm: list[float] = []
    right_status: list[str] = []
    left_status: list[str] = []

    for row in rows:
        t = float(row["t_monotonic"])
        if t < t_start or t > t_end:
            continue
        time_s.append(t - t_start)
        sequence.append(int(row.get("sequence", -1)))
        right_distance_cm.append(_float_or_nan(row.get("right_distance_cm")))
        left_distance_cm.append(_float_or_nan(row.get("left_distance_cm")))
        right_status.append(row.get("right_status", "missing"))
        left_status.append(row.get("left_status", "missing"))

    return {
        "time_s": np.array(time_s, dtype=float),
        "sequence": np.array(sequence, dtype=np.int64),
        "right_distance_cm": np.array(right_distance_cm, dtype=float),
        "left_distance_cm": np.array(left_distance_cm, dtype=float),
        "right_status": np.array(right_status),
        "left_status": np.array(left_status),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "trials.csv"

    allowed_collectors = normalize_filter(args.collector)
    allowed_gestures = normalize_filter(args.gesture)

    combined: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    gesture_counts: dict[str, int] = {}
    collector_counts: dict[str, int] = {}

    for session_arg in args.sessions:
        session_dir = Path(session_arg).expanduser().resolve()
        metadata_path = session_dir / "session_metadata.json"
        if not metadata_path.exists():
            skipped.append({"session_dir": str(session_dir), "reason": "missing session_metadata.json"})
            continue

        metadata = json.loads(metadata_path.read_text())
        collector = str(metadata.get("collector", ""))
        if allowed_collectors and collector not in allowed_collectors:
            continue

        trial_rows = read_csv_rows(session_dir / "trials.csv")
        if not trial_rows:
            skipped.append({"session_dir": str(session_dir), "reason": "missing or empty trials.csv"})
            continue

        imu_rows = read_csv_rows(session_dir / "imu.csv")
        mmwave_rows = read_csv_rows(session_dir / "mmwave.csv")
        uwb_rows = read_csv_rows(session_dir / "uwb.csv")

        for trial_row in trial_rows:
            gesture = trial_row["gesture"]
            if allowed_gestures and gesture not in allowed_gestures:
                continue

            accepted = trial_row.get("accepted", "1") == "1"
            if not accepted and not args.include_rejected:
                continue

            t_start = float(trial_row["t_start"])
            t_end = float(trial_row["t_end"])
            trial_index = int(trial_row["trial_index"])
            duration_s = max(t_end - t_start, 1e-9)

            imu_segment = slice_imu(imu_rows, t_start, t_end) if imu_rows else None
            mmwave_segment = slice_mmwave(mmwave_rows, t_start, t_end) if mmwave_rows else None
            uwb_segment = slice_uwb(uwb_rows, t_start, t_end) if uwb_rows else None

            imu_count = len(imu_segment["time_s"]) if imu_segment is not None else 0
            mmwave_count = len(mmwave_segment["time_s"]) if mmwave_segment is not None else 0
            uwb_count = len(uwb_segment["time_s"]) if uwb_segment is not None else 0
            if imu_count == 0 and mmwave_count == 0 and uwb_count == 0:
                skipped.append(
                    {
                        "session_dir": str(session_dir),
                        "trial_index": trial_index,
                        "reason": "no imu, mmwave, or uwb samples fell inside [t_start, t_end]",
                    }
                )
                continue

            trial_name = f"{safe_label(collector)}_{safe_label(gesture)}_{trial_index:03d}"
            npz_path = trials_dir / f"{trial_name}.npz"

            payload: dict[str, np.ndarray] = {
                "collector": np.array(collector),
                "gesture": np.array(gesture),
                "trial_index": np.array(trial_index),
                "t_start": np.array(t_start),
                "t_end": np.array(t_end),
                "source_session": np.array(str(session_dir)),
            }
            if imu_segment is not None:
                payload.update({f"imu_{key}": value for key, value in imu_segment.items()})
            if mmwave_segment is not None:
                payload.update({f"mmwave_{key}": value for key, value in mmwave_segment.items()})
            if uwb_segment is not None:
                payload.update({f"uwb_{key}": value for key, value in uwb_segment.items()})

            np.savez_compressed(npz_path, **payload)

            row = {
                "dataset_name": output_dir.name,
                "source_session": str(session_dir),
                "collector": collector,
                "gesture": gesture,
                "trial_index": trial_index,
                "t_start": f"{t_start:.6f}",
                "t_end": f"{t_end:.6f}",
                "duration_s": f"{duration_s:.6f}",
                "imu_sample_count": imu_count,
                "imu_mean_rate_hz": f"{imu_count / duration_s:.2f}" if imu_count else "0",
                "mmwave_frame_count": mmwave_count,
                "mmwave_mean_rate_hz": f"{mmwave_count / duration_s:.2f}" if mmwave_count else "0",
                "uwb_round_count": uwb_count,
                "uwb_mean_rate_hz": f"{uwb_count / duration_s:.2f}" if uwb_count else "0",
                "npz_path": str(npz_path),
                "session_dir": str(session_dir),
            }
            combined.append(row)
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
            collector_counts[collector] = collector_counts.get(collector, 0) + 1

    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(combined)

    dataset_metadata = {
        "dataset_name": output_dir.name,
        "combined_from": [str(Path(item).expanduser().resolve()) for item in args.sessions],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "row_count": len(combined),
        "gesture_counts": gesture_counts,
        "collector_counts": collector_counts,
        "collector_filter": sorted(allowed_collectors) if allowed_collectors else None,
        "gesture_filter": sorted(allowed_gestures) if allowed_gestures else None,
        "include_rejected": args.include_rejected,
        "skipped": skipped,
    }
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(dataset_metadata, indent=2, sort_keys=True))
    if not combined:
        print("No trials were combined.", file=sys.stderr)
        return 1
    print(f"Combined manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
