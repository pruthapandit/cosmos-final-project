#!/usr/bin/env python3
"""Semi-live gesture prediction: press Enter, record ~2s from whichever
sensors the loaded model needs, extract the same features train_gesture.py
uses, and predict.

This is not continuous/streaming recognition -- it still relies on a known
recording window (like collect.py's trials) rather than solving gesture
segmentation in an unbroken stream. See GESTURE_PIPELINE.md for why.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import train_gesture as tg
from sensors.imu_reader import ImuReader
from sensors.mmwave_reader import MmwaveReader
from sensors.uwb_reader import UwbReader


def build_imu_dict(samples: list[tuple[float, dict[str, Any]]], t_start: float) -> dict[str, np.ndarray]:
    time_s: list[float] = []
    columns: dict[str, list[float]] = {name: [] for name in ("ax", "ay", "az", "gx", "gy", "gz")}
    for t, fields in samples:
        time_s.append(t - t_start)
        for name in columns:
            columns[name].append(float(fields[name]))
    result = {"imu_time_s": np.array(time_s, dtype=float)}
    result.update({f"imu_{name}": np.array(values, dtype=float) for name, values in columns.items()})
    return result


def build_mmwave_dict(samples: list[tuple[float, dict[str, Any]]], t_start: float) -> dict[str, np.ndarray]:
    time_s: list[float] = []
    profiles: list[list[float]] = []
    for t, fields in samples:
        time_s.append(t - t_start)
        profiles.append(json.loads(fields["profile_json"]))
    if not profiles:
        range_profile = np.empty((0, 0), dtype=float)
    else:
        bin_count = max(len(profile) for profile in profiles)
        range_profile = np.zeros((len(profiles), bin_count), dtype=float)
        for index, profile in enumerate(profiles):
            range_profile[index, : len(profile)] = profile
    return {
        "mmwave_time_s": np.array(time_s, dtype=float),
        "mmwave_range_profile": range_profile,
    }


def build_uwb_dict(samples: list[tuple[float, dict[str, Any]]], t_start: float) -> dict[str, np.ndarray]:
    time_s: list[float] = []
    right_distance_cm: list[float] = []
    left_distance_cm: list[float] = []
    for t, fields in samples:
        time_s.append(t - t_start)
        right_distance_cm.append(float(fields.get("right_distance_cm") or "nan"))
        left_distance_cm.append(float(fields.get("left_distance_cm") or "nan"))
    return {
        "uwb_time_s": np.array(time_s, dtype=float),
        "uwb_right_distance_cm": np.array(right_distance_cm, dtype=float),
        "uwb_left_distance_cm": np.array(left_distance_cm, dtype=float),
    }


def build_readers(sensor_names: list[str], args: argparse.Namespace) -> dict[str, Any]:
    readers: dict[str, Any] = {}
    if "imu" in sensor_names:
        readers["imu"] = ImuReader(port=args.imu_port, baud=args.imu_baud, max_rate_hz=args.imu_rate_hz)
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


def feature_args_from_payload(payload: dict[str, Any], sensors: list[str]) -> argparse.Namespace:
    return argparse.Namespace(
        sensors=sensors,
        imu_trajectory_points=payload.get("imu_trajectory_points", 10),
        mmwave_time_bins=payload.get("mmwave_time_bins", 16),
        mmwave_range_bins=payload.get("mmwave_range_bins", 24),
        mmwave_min_range_m=payload.get("mmwave_min_range_m", 0.15),
        mmwave_max_range_m=payload.get("mmwave_max_range_m", 2.0),
        mmwave_bin_spacing_m=payload.get("mmwave_bin_spacing_m", tg.MMWAVE_BIN_SPACING_M),
        uwb_trajectory_points=payload.get("uwb_trajectory_points", 10),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semi-live gesture prediction from a trained model.")
    parser.add_argument("--model", required=True, help="Path to a .joblib model saved by train_gesture.py")
    parser.add_argument("--trial-seconds", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=3, help="How many ranked predictions to print.")
    parser.add_argument(
        "--reject-threshold", type=float, default=0.4,
        help="If the top class's probability is below this, report 'None' instead of forcing a "
        "guess -- the model is a closed-set classifier and will otherwise always name one of its "
        "trained gestures even for a completely unrelated motion. 0 disables rejection.",
    )

    parser.add_argument("--imu-port", default=None)
    parser.add_argument("--imu-baud", type=int, default=115200)
    parser.add_argument("--imu-rate-hz", type=float, default=25.0)

    parser.add_argument("--mmwave-port", default=None)
    parser.add_argument("--mmwave-cfg", type=Path, default=None)
    parser.add_argument("--mmwave-baud", type=int, default=115200)
    parser.add_argument("--mmwave-frame-timeout", type=float, default=5.0)
    parser.add_argument("--mmwave-use-cfg-baud-rate", action="store_true")

    parser.add_argument("--uwb-controller-port", default=None)
    parser.add_argument("--uwb-right-port", default=None)
    parser.add_argument("--uwb-left-port", default=None)
    parser.add_argument("--uwb-tools-root", type=Path, default=Path(r"C:\Users\Prutha Pandit\UWB_lab\uwb-qorvo-tools"))
    parser.add_argument("--uwb-channel", type=int, default=5)
    parser.add_argument("--uwb-preamble-idx", type=int, default=12)
    parser.add_argument("--uwb-session", type=int, default=42)
    parser.add_argument("--uwb-slot-span", type=int, default=2400)
    parser.add_argument("--uwb-slots-per-rr", type=int, default=25)
    parser.add_argument("--uwb-ranging-span", type=int, default=50)
    parser.add_argument("--uwb-startup-delay", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import joblib

    payload = joblib.load(Path(args.model).expanduser().resolve())
    model = payload["model"]
    sensors = payload["sensors"]
    labels_order = payload.get("labels_order")
    print(f"Loaded model: sensors={'+'.join(sensors)}  classifier={payload.get('classifier_label', '?')}  "
          f"trained accuracy={payload.get('accuracy', float('nan')):.3f}")

    if "imu" in sensors and not args.imu_port:
        raise SystemExit("--imu-port is required: this model uses imu")
    if "mmwave" in sensors and (not args.mmwave_port or not args.mmwave_cfg):
        raise SystemExit("--mmwave-port and --mmwave-cfg are required: this model uses mmwave")
    if "uwb" in sensors and not (args.uwb_controller_port and args.uwb_right_port and args.uwb_left_port):
        raise SystemExit("--uwb-controller-port, --uwb-right-port, and --uwb-left-port are required: this model uses uwb")

    feature_args = feature_args_from_payload(payload, sensors)
    readers = build_readers(sensors, args)

    started_reader_names: list[str] = []
    try:
        for name, reader in readers.items():
            reader.start()
            started_reader_names.append(name)

        while True:
            print(f"\nPress Enter to record a gesture ({args.trial_seconds:g}s), or Ctrl+C to quit...")
            input()

            # Discard whatever queued up while we were idle so the window
            # starts clean, exactly like collect.py's per-trial drain.
            for reader in readers.values():
                reader.drain()

            t_start = time.monotonic()
            end_at = t_start + args.trial_seconds
            while time.monotonic() < end_at:
                remaining = end_at - time.monotonic()
                print(f"  recording... {remaining:4.1f}s left", end="\r")
                time.sleep(0.05)
            t_end = time.monotonic()
            print(f"\n  captured {t_end - t_start:.2f}s")

            data: dict[str, Any] = {}
            for name, reader in readers.items():
                samples, errors = reader.drain()
                for err in errors:
                    print(f"  [{name}] ERROR: {err}")
                samples = [(t, f) for t, f in samples if t_start <= t <= t_end]
                print(f"  [{name}] {len(samples)} samples")
                if name == "imu":
                    data.update(build_imu_dict(samples, t_start))
                elif name == "mmwave":
                    data.update(build_mmwave_dict(samples, t_start))
                elif name == "uwb":
                    data.update(build_uwb_dict(samples, t_start))

            features, _ = tg.extract_features(data, feature_args)
            X = np.asarray([features], dtype=float)
            proba = model.predict_proba(X)[0]
            classes = model.classes_
            top_idx = int(np.argmax(proba))
            top_confidence = float(proba[top_idx])
            raw_prediction = str(classes[top_idx])

            if args.reject_threshold and top_confidence < args.reject_threshold:
                prediction = "None"
                print(f"\n  >>> Predicted: None  (best guess {raw_prediction} only "
                      f"{top_confidence * 100:.0f}% confident -- below --reject-threshold "
                      f"{args.reject_threshold:.2f})")
            else:
                prediction = raw_prediction
                print(f"\n  >>> Predicted: {prediction}")

            order = np.argsort(proba)[::-1][: args.top_k]
            for i in order:
                print(f"      {classes[i]:22s} {proba[i] * 100:5.1f}%")
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        for name in reversed(started_reader_names):
            readers[name].stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
