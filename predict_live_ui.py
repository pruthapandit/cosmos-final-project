#!/usr/bin/env python3
"""Browser UI for the button-triggered 2-second record-and-predict flow
(predict_live.py's logic, with a nicer, screen-recordable dashboard instead
of a terminal loop).

Click "Record" in the browser -> POST /record blocks for --window-seconds
while the ring countdown animates client-side; meanwhile the browser polls
GET /live every ~150ms so the charts actually draw the signal as it's
captured, not just a snapshot revealed at the end. A single background
thread continuously drains the sensor readers into a rolling LiveBuffer --
the sole consumer of reader.drain() -- so /live (polling) and /record
(final classification) both read from that same buffer instead of racing
each other over the raw queues.
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

HTML_PATH = Path(__file__).with_name("predict_live_ui.html")
BUFFER_SECONDS = 5.0  # rolling history kept per sensor, bounds memory


class LiveBuffer:
    """Continuously-fed rolling (t, fields) history per sensor.

    One background thread is the only thing that ever calls reader.drain();
    everything else (the /live poll, the final /record classification)
    reads a windowed slice out of here instead, so nothing races over the
    same queue.
    """

    def __init__(self, sensor_names: list[str]) -> None:
        self._lock = threading.Lock()
        self._buffers: dict[str, deque] = {name: deque() for name in sensor_names}

    def ingest(self, name: str, samples: list[tuple[float, dict[str, Any]]]) -> None:
        if not samples:
            return
        with self._lock:
            buf = self._buffers[name]
            buf.extend(samples)
            cutoff = time.monotonic() - BUFFER_SECONDS
            while buf and buf[0][0] < cutoff:
                buf.popleft()

    def window(self, name: str, t_start: float, t_end: float) -> list[tuple[float, dict[str, Any]]]:
        with self._lock:
            return [(t, f) for t, f in self._buffers[name] if t_start <= t <= t_end]


def buffering_loop(readers: dict[str, Any], live_buffer: LiveBuffer, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        for name, reader in readers.items():
            samples, _errors = reader.drain()
            live_buffer.ingest(name, samples)
        time.sleep(0.08)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser UI for button-triggered gesture prediction.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--trial-seconds", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--web-port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")

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


def windowed_chart_data(
    live_buffer: LiveBuffer,
    sensor_names: list[str],
    feature_args: argparse.Namespace,
    t_start: float,
    t_end: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Slice [t_start, t_end] out of the live buffer, returning both the
    feature-extraction-ready `data` dict and the chart-friendly series."""
    data: dict[str, Any] = {}
    gyro_series: list[float] = []
    uwb_right_series: list[float] = []
    uwb_left_series: list[float] = []
    mmwave_profile: list[float] = []

    for name in sensor_names:
        samples = live_buffer.window(name, t_start, t_end)
        if name == "imu":
            data.update(build_imu_dict(samples, t_start))
            gx = data.get("imu_gx", np.array([]))
            gy = data.get("imu_gy", np.array([]))
            gz = data.get("imu_gz", np.array([]))
            if gx.size:
                gyro_series = list(np.sqrt(gx**2 + gy**2 + gz**2).astype(float))
        elif name == "mmwave":
            data.update(build_mmwave_dict(samples, t_start))
            profile = data.get("mmwave_range_profile")
            if profile is not None and profile.size:
                cropped = tg.crop_mmwave_range(
                    profile, feature_args.mmwave_min_range_m,
                    feature_args.mmwave_max_range_m, feature_args.mmwave_bin_spacing_m,
                )
                if cropped.size:
                    mmwave_profile = [float(v) for v in cropped[-1]]
        elif name == "uwb":
            data.update(build_uwb_dict(samples, t_start))
            right = data.get("uwb_right_distance_cm", np.array([]))
            left = data.get("uwb_left_distance_cm", np.array([]))
            uwb_right_series = [v for v in right.astype(float) if np.isfinite(v)]
            uwb_left_series = [v for v in left.astype(float) if np.isfinite(v)]

    charts = {
        "gyro_series": gyro_series,
        "uwb_right_series": uwb_right_series,
        "uwb_left_series": uwb_left_series,
        "mmwave_profile": mmwave_profile,
    }
    return data, charts


def do_one_recording(
    live_buffer: LiveBuffer,
    sensor_names: list[str],
    model: Any,
    feature_args: argparse.Namespace,
    trial_seconds: float,
    top_k: int,
    history: deque[str],
) -> dict[str, Any]:
    t_start = time.monotonic()
    time.sleep(trial_seconds)
    t_end = time.monotonic()

    data, charts = windowed_chart_data(live_buffer, sensor_names, feature_args, t_start, t_end)

    features, _ = tg.extract_features(data, feature_args)
    X = np.asarray([features], dtype=float)
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    order = np.argsort(proba)[::-1][:top_k]
    top_k_list = [(str(classes[i]), float(proba[i])) for i in order]
    prediction = top_k_list[0][0]

    history.append(prediction)

    return {
        "prediction": prediction,
        "confidence": top_k_list[0][1],
        "top3": top_k_list,
        "history": list(history),
        **charts,
    }


def make_handler(
    live_buffer: LiveBuffer,
    model: Any,
    feature_args: argparse.Namespace,
    args: argparse.Namespace,
    sensors: list[str],
    accuracy: float,
    history: deque[str],
    record_lock: threading.Lock,
):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args_: Any) -> None:
            pass

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                body = HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/info":
                self._send_json({
                    "sensors": sensors,
                    "accuracy": accuracy,
                    "window_seconds": args.trial_seconds,
                })
                return

            if self.path == "/live":
                # Polled every ~150ms by the frontend during the countdown,
                # so the charts draw the signal as it's actually captured
                # instead of only revealing it once /record finishes.
                now = time.monotonic()
                _data, charts = windowed_chart_data(
                    live_buffer, sensors, feature_args, now - args.trial_seconds, now
                )
                self._send_json(charts)
                return

            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:
            if self.path == "/record":
                # Only one capture at a time so two clicks can't overlap
                # and confuse whose window a classification belongs to.
                if record_lock.locked():
                    self._send_json({"error": "already recording"})
                    return
                with record_lock:
                    result = do_one_recording(
                        live_buffer, sensors, model, feature_args, args.trial_seconds, args.top_k, history
                    )
                self._send_json(result)
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
    accuracy = float(payload.get("accuracy", float("nan")))
    print(f"Loaded model: sensors={'+'.join(sensors)}  classifier={payload.get('classifier_label', '?')}  "
          f"trained accuracy={accuracy:.3f}")

    if "imu" in sensors and not args.imu_port:
        raise SystemExit("--imu-port is required: this model uses imu")
    if "mmwave" in sensors and (not args.mmwave_port or not args.mmwave_cfg):
        raise SystemExit("--mmwave-port and --mmwave-cfg are required: this model uses mmwave")
    if "uwb" in sensors and not (args.uwb_controller_port and args.uwb_right_port and args.uwb_left_port):
        raise SystemExit("--uwb-controller-port, --uwb-right-port, and --uwb-left-port are required: this model uses uwb")

    feature_args = feature_args_from_payload(payload, sensors)
    readers = build_readers(sensors, args)
    history: deque[str] = deque(maxlen=10)
    record_lock = threading.Lock()
    live_buffer = LiveBuffer(sensors)
    stop_event = threading.Event()

    for reader in readers.values():
        reader.start()

    buffer_thread = threading.Thread(
        target=buffering_loop, args=(readers, live_buffer, stop_event), daemon=True
    )
    buffer_thread.start()

    server = ThreadingHTTPServer(
        ("127.0.0.1", args.web_port),
        make_handler(live_buffer, model, feature_args, args, sensors, accuracy, history, record_lock),
    )
    url = f"http://127.0.0.1:{args.web_port}"
    print(f"\nOpen {url} and click Record.  (Ctrl+C to stop)\n")
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
