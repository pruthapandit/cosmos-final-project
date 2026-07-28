#!/usr/bin/env python3
"""Browser dashboard for predict_stream.py's continuous gesture prediction.

Same two-tier-confidence sliding-window logic as predict_stream.py, but
instead of printing to a terminal, it runs a small local HTTP server (no new
dependencies -- stdlib http.server only) and pushes live prediction +
sensor-signal updates to predict_stream_ui.html over Server-Sent Events.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

HTML_PATH = Path(__file__).with_name("predict_stream_ui.html")
CHART_HISTORY = 120  # points kept for the scrolling charts (~ CHART_HISTORY * tick-seconds of history)


class SharedState:
    def __init__(self, sensors: list[str], accuracy: float) -> None:
        self._lock = threading.Lock()
        self.sensors = sensors
        self.accuracy = accuracy
        self.prediction = "Idle"
        self.confidence = 0.0
        self.top3: list[tuple[str, float]] = []
        self.history: deque[str] = deque(maxlen=10)
        self.gyro_series: deque[float] = deque(maxlen=CHART_HISTORY)
        self.uwb_right_series: deque[float] = deque(maxlen=CHART_HISTORY)
        self.uwb_left_series: deque[float] = deque(maxlen=CHART_HISTORY)
        self.mmwave_profile: list[float] = []
        self.history.append("Idle")

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def push_gesture(self, gesture: str) -> None:
        with self._lock:
            self.history.append(gesture)

    def extend_series(self, gyro: list[float], uwb_right: list[float], uwb_left: list[float]) -> None:
        with self._lock:
            self.gyro_series.extend(gyro)
            self.uwb_right_series.extend(uwb_right)
            self.uwb_left_series.extend(uwb_left)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sensors": self.sensors,
                "model_accuracy": self.accuracy,
                "prediction": self.prediction,
                "confidence": self.confidence,
                "top3": list(self.top3),
                "history": list(self.history),
                "gyro_series": [v for v in self.gyro_series if np.isfinite(v)],
                "uwb_right_series": [v for v in self.uwb_right_series if np.isfinite(v)],
                "uwb_left_series": [v for v in self.uwb_left_series if np.isfinite(v)],
                "mmwave_profile": self.mmwave_profile,
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser dashboard for continuous gesture prediction.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--tick-seconds", type=float, default=0.3)
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--switch-to-gesture-votes", type=int, default=2)
    parser.add_argument("--switch-to-idle-votes", type=int, default=2)
    parser.add_argument("--web-port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab.")

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


def prediction_loop(
    readers: dict[str, Any],
    model: Any,
    feature_args: argparse.Namespace,
    state: SharedState,
    args: argparse.Namespace,
    stop_event: threading.Event,
) -> None:
    buffers: dict[str, deque] = {name: deque() for name in readers}
    displayed = "Idle"
    pending_candidate: str | None = None
    pending_count = 0

    while not stop_event.is_set():
        time.sleep(args.tick_seconds)
        now = time.monotonic()
        window_start = now - args.window_seconds

        for name, reader in readers.items():
            samples, _errors = reader.drain()
            buffers[name].extend(samples)
            while buffers[name] and buffers[name][0][0] < window_start - 1.0:
                buffers[name].popleft()

        data: dict[str, Any] = {}
        for name in readers:
            windowed = [(t, f) for t, f in buffers[name] if window_start <= t <= now]
            if name == "imu":
                data.update(build_imu_dict(windowed, window_start))
                gx = data.get("imu_gx", np.array([]))
                gy = data.get("imu_gy", np.array([]))
                gz = data.get("imu_gz", np.array([]))
                if gx.size:
                    mag = np.sqrt(gx**2 + gy**2 + gz**2)
                    # Sample down to a handful of points per tick so the chart
                    # scrolls smoothly without accumulating unbounded detail.
                    take = mag[-max(1, mag.size // 5):]
                    state.extend_series(list(take.astype(float)), [], [])
            elif name == "mmwave":
                data.update(build_mmwave_dict(windowed, window_start))
                profile = data.get("mmwave_range_profile")
                if profile is not None and profile.size:
                    cropped = tg.crop_mmwave_range(
                        profile, feature_args.mmwave_min_range_m,
                        feature_args.mmwave_max_range_m, feature_args.mmwave_bin_spacing_m,
                    )
                    if cropped.size:
                        state.update(mmwave_profile=[float(v) for v in cropped[-1]])
            elif name == "uwb":
                data.update(build_uwb_dict(windowed, window_start))
                right = data.get("uwb_right_distance_cm", np.array([]))
                left = data.get("uwb_left_distance_cm", np.array([]))
                if right.size or left.size:
                    take_r = right[-max(1, right.size // 5):] if right.size else np.array([])
                    take_l = left[-max(1, left.size // 5):] if left.size else np.array([])
                    state.extend_series([], list(take_r.astype(float)), list(take_l.astype(float)))

        features, _ = tg.extract_features(data, feature_args)
        X = np.asarray([features], dtype=float)
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        top_idx = np.argsort(proba)[::-1][:3]
        top3 = [(str(classes[i]), float(proba[i])) for i in top_idx]
        raw_prediction = top3[0][0]
        top_confidence = top3[0][1]
        state.update(top3=top3)

        # Below --confidence-threshold: not a transient blip to ignore, the
        # model doesn't confidently match ANY trained class -- report "None"
        # (same quick/safe persistence as Idle) instead of silently freezing
        # on whatever real gesture was last displayed.
        prediction = "None" if top_confidence < args.confidence_threshold else raw_prediction

        if prediction == displayed:
            pending_candidate = None
            pending_count = 0
        elif prediction != "None" and top_confidence >= args.high_confidence_threshold:
            displayed = prediction
            pending_candidate = None
            pending_count = 0
            state.update(prediction=displayed, confidence=top_confidence)
            state.push_gesture(displayed)
        else:
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
                state.update(prediction=displayed, confidence=top_confidence)
                state.push_gesture(displayed)


def make_handler(state: SharedState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args_: Any) -> None:
            pass  # keep the console quiet during a demo

        def do_GET(self) -> None:
            if self.path == "/":
                body = HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    while True:
                        payload = json.dumps(state.snapshot())
                        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        time.sleep(0.25)
                except (BrokenPipeError, ConnectionResetError):
                    return
                return

            self.send_response(404)
            self.end_headers()

    return Handler


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
    state = SharedState(sensors=sensors, accuracy=float(payload.get("accuracy", float("nan"))))
    stop_event = threading.Event()

    for reader in readers.values():
        reader.start()

    worker = threading.Thread(
        target=prediction_loop, args=(readers, model, feature_args, state, args, stop_event), daemon=True
    )
    worker.start()

    server = ThreadingHTTPServer(("127.0.0.1", args.web_port), make_handler(state))
    url = f"http://127.0.0.1:{args.web_port}"
    print(f"\nDashboard running at {url}  (Ctrl+C to stop)\n")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        stop_event.set()
        server.shutdown()
        for reader in readers.values():
            reader.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
