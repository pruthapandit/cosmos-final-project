#!/usr/bin/env python3
"""Continuous gesture prediction: no button press, just a live rolling window.

Every --tick-seconds, classifies the last --window-seconds of buffered
sensor data (same features as train_gesture.py) and prints a temporally
smoothed prediction (majority vote over the last few ticks, to avoid
flickering mid-gesture). Relies on the model having a trained "Idle" class
to naturally absorb periods where nothing is happening -- there is no
separate motion-onset detector.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

import train_gesture as tg
from predict_live import (
    build_imu_dict,
    build_mmwave_dict,
    build_uwb_dict,
    build_readers,
    feature_args_from_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous sliding-window gesture prediction.")
    parser.add_argument("--model", required=True, help="Path to a .joblib model saved by train_gesture.py")
    parser.add_argument("--window-seconds", type=float, default=2.0,
                         help="How much recent history to classify each tick (should match training trial length).")
    parser.add_argument("--tick-seconds", type=float, default=0.2,
                         help="How often to re-classify. Shorter = more chances for a sliding window "
                         "to land well-aligned on a brief, one-shot (discrete) gesture.")
    parser.add_argument("--confidence-threshold", type=float, default=0.4,
                         help="Ignore a tick's prediction if the model's top-class probability is "
                         "below this -- transitional motion between gestures tends to produce "
                         "low-confidence, ambiguous predictions rather than a clean match, so this "
                         "filters most of it out before it ever reaches the vote.")
    parser.add_argument("--high-confidence-threshold", type=float, default=0.65,
                         help="A single tick at or above this confidence switches immediately, "
                         "bypassing --switch-to-gesture-votes. Needed for discrete one-shot gestures "
                         "(Pull, Push, Clockwise, One-Arm Boxing, ...) which only happen once -- unlike "
                         "continuous/repetitive gestures (Bye-Bye, Soli, Making Fist and Open, Palm "
                         "Up-Down), a discrete gesture may only produce one or two well-aligned windows "
                         "before the sliding view moves past it, so requiring several consecutive "
                         "confident ticks can miss it entirely.")
    parser.add_argument("--switch-to-gesture-votes", type=int, default=2,
                         help="Consecutive moderately-confident (>= --confidence-threshold but below "
                         "--high-confidence-threshold) agreeing ticks required to switch the displayed "
                         "prediction to a new (non-Idle) gesture.")
    parser.add_argument("--switch-to-idle-votes", type=int, default=2,
                         help="Consecutive confident agreeing ticks required to fall back to Idle. "
                         "Kept lower than --switch-to-gesture-votes so the system resets to the safe "
                         "default quickly once motion stops.")

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
    print(f"Loaded model: sensors={'+'.join(sensors)}  classifier={payload.get('classifier_label', '?')}  "
          f"trained accuracy={payload.get('accuracy', float('nan')):.3f}")
    if "Idle" not in payload.get("labels_order", []):
        print("WARNING: this model has no 'Idle' class -- it will confidently guess a real gesture "
              "even when nothing is happening.")

    if "imu" in sensors and not args.imu_port:
        raise SystemExit("--imu-port is required: this model uses imu")
    if "mmwave" in sensors and (not args.mmwave_port or not args.mmwave_cfg):
        raise SystemExit("--mmwave-port and --mmwave-cfg are required: this model uses mmwave")
    if "uwb" in sensors and not (args.uwb_controller_port and args.uwb_right_port and args.uwb_left_port):
        raise SystemExit("--uwb-controller-port, --uwb-right-port, and --uwb-left-port are required: this model uses uwb")

    feature_args = feature_args_from_payload(payload, sensors)
    readers = build_readers(sensors, args)

    # Rolling per-sensor buffers of every (t, fields) sample seen, trimmed
    # each tick to just past the window so memory doesn't grow unbounded.
    buffers: dict[str, deque] = {name: deque() for name in readers}
    displayed = "Idle"
    pending_candidate: str | None = None
    pending_count = 0

    for reader in readers.values():
        reader.start()

    print(f"\nStreaming... (Ctrl+C to stop). Classifying the last {args.window_seconds:g}s every "
          f"{args.tick_seconds:g}s. confidence<{args.confidence_threshold:.2f} reports 'None' "
          f"(untrained/unrecognized motion); confidence>={args.confidence_threshold:.2f} needs "
          f"{args.switch_to_gesture_votes} consecutive ticks to switch to a gesture "
          f"({args.switch_to_idle_votes} to fall back to Idle/None); a single tick "
          f">={args.high_confidence_threshold:.2f} switches immediately.\n")

    try:
        while True:
            time.sleep(args.tick_seconds)
            now = time.monotonic()
            window_start = now - args.window_seconds

            for name, reader in readers.items():
                samples, errors = reader.drain()
                for err in errors:
                    print(f"\n[{name}] ERROR: {err}")
                buffers[name].extend(samples)
                while buffers[name] and buffers[name][0][0] < window_start - 1.0:
                    buffers[name].popleft()

            data: dict[str, Any] = {}
            for name in readers:
                windowed = [(t, f) for t, f in buffers[name] if window_start <= t <= now]
                if name == "imu":
                    data.update(build_imu_dict(windowed, window_start))
                elif name == "mmwave":
                    data.update(build_mmwave_dict(windowed, window_start))
                elif name == "uwb":
                    data.update(build_uwb_dict(windowed, window_start))

            features, _ = tg.extract_features(data, feature_args)
            X = np.asarray([features], dtype=float)
            proba = model.predict_proba(X)[0]
            classes = model.classes_
            top_idx = np.argmax(proba)
            raw_prediction = str(classes[top_idx])
            top_confidence = float(proba[top_idx])

            # Below --confidence-threshold: this isn't a transient blip to
            # ignore, it's the model telling us the input doesn't confidently
            # match ANY trained class -- likely an untrained/random motion.
            # Treat "None" as its own candidate (same quick/safe persistence
            # as Idle) rather than silently freezing on the last real gesture.
            prediction = "None" if top_confidence < args.confidence_threshold else raw_prediction

            if prediction == displayed:
                pending_candidate = None
                pending_count = 0
            elif prediction != "None" and top_confidence >= args.high_confidence_threshold:
                # Fast path: one very confident tick is enough. Needed for
                # discrete one-shot gestures, which may only ever produce a
                # single well-aligned window before the slide moves past them.
                displayed = prediction
                pending_candidate = None
                pending_count = 0
                print(f"  >>> {displayed}  ({top_confidence * 100:.0f}% this tick, fast path)")
            else:
                # Slow path: moderate confidence (or a sustained "None") needs
                # to repeat a few times before we commit, which is what
                # filters out transitional motion (it rarely stays
                # confidently pointed at one class for long).
                if prediction == pending_candidate:
                    pending_count += 1
                else:
                    pending_candidate = prediction
                    pending_count = 1
                required = args.switch_to_idle_votes if prediction in ("Idle", "None") else args.switch_to_gesture_votes
                if pending_count >= required:
                    displayed = prediction
                    pending_candidate = None
                    pending_count = 0
                    print(f"  >>> {displayed}  ({top_confidence * 100:.0f}% this tick)")
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        for reader in readers.values():
            reader.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
