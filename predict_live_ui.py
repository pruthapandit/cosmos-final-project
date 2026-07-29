#!/usr/bin/env python3
"""Browser UI for the button-triggered 2-second record-and-predict flow
(predict_live.py's logic, with a nicer, screen-recordable dashboard instead
of a terminal loop).

Click "Record" in the browser -> POST /record drains stale samples, sleeps
--window-seconds while the ring countdown animates client-side (a plain
CSS/JS timer, no server round-trips), drains the real window, classifies,
and returns the result. No live chart polling -- keeping this to one
request per capture keeps the hot path (drain/sleep/drain/classify) as fast
as possible.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser UI for button-triggered gesture prediction.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--trial-seconds", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--reject-threshold", type=float, default=0.45,
        help="If the top class's probability is below this, report 'None' instead of forcing a "
        "guess. 0 disables rejection.",
    )
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
    parser.add_argument(
        "--uwb-plot-dir",
        type=Path,
        default=Path("captures") / "uwb",
        help="Folder where each UWB-enabled capture's wrist-distance plot is saved.",
    )
    return parser.parse_args()


def save_uwb_plot(
    data: dict[str, Any], output_dir: Path, prediction: str, confidence: float
) -> Path | None:
    """Save the UWB samples from one capture as a timestamped PNG."""
    time_s = np.asarray(data.get("uwb_time_s", []), dtype=float)
    right = np.asarray(data.get("uwb_right_distance_cm", []), dtype=float)
    left = np.asarray(data.get("uwb_left_distance_cm", []), dtype=float)
    count = min(len(time_s), len(right), len(left))
    if count == 0:
        return None

    time_s, right, left = time_s[:count], right[:count], left[:count]
    right_valid = np.isfinite(time_s) & np.isfinite(right)
    left_valid = np.isfinite(time_s) & np.isfinite(left)
    if not right_valid.any() and not left_valid.any():
        return None

    # Keep the web server's plotting independent of a desktop GUI backend.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"uwb_capture_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}.png"
    output_path = output_dir / filename

    figure, axis = plt.subplots(figsize=(8, 4.5))
    if right_valid.any():
        axis.plot(time_s[right_valid], right[right_valid], "o-", label="Right wrist", color="#4f8cff")
    if left_valid.any():
        axis.plot(time_s[left_valid], left[left_valid], "o-", label="Left wrist", color="#f59e0b")
    axis.set_title(f"UWB capture — {prediction} ({confidence:.1%})")
    axis.set_xlabel("Time since capture start (s)")
    axis.set_ylabel("Distance (cm)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def do_one_recording(
    readers: dict[str, Any],
    model: Any,
    feature_args: argparse.Namespace,
    trial_seconds: float,
    top_k: int,
    reject_threshold: float,
    history: deque[str],
    uwb_plot_dir: Path | None,
) -> dict[str, Any]:
    # Discard whatever queued up while idle so the window starts clean.
    for reader in readers.values():
        reader.drain()

    t_start = time.monotonic()
    time.sleep(trial_seconds)
    t_end = time.monotonic()

    data: dict[str, Any] = {}
    uwb_right_series: list[float] = []
    uwb_left_series: list[float] = []
    for name, reader in readers.items():
        samples, _errors = reader.drain()
        samples = [(t, f) for t, f in samples if t_start <= t <= t_end]
        if name == "imu":
            data.update(build_imu_dict(samples, t_start))
        elif name == "mmwave":
            data.update(build_mmwave_dict(samples, t_start))
        elif name == "uwb":
            data.update(build_uwb_dict(samples, t_start))
            right = data.get("uwb_right_distance_cm", np.array([]))
            left = data.get("uwb_left_distance_cm", np.array([]))
            uwb_right_series = [v for v in right.astype(float) if np.isfinite(v)]
            uwb_left_series = [v for v in left.astype(float) if np.isfinite(v)]

    features, _ = tg.extract_features(data, feature_args)
    X = np.asarray([features], dtype=float)
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    order = np.argsort(proba)[::-1][:top_k]
    top_k_list = [(str(classes[i]), float(proba[i])) for i in order]
    top_confidence = top_k_list[0][1]

    # Closed-set classifier: it will always name one of its trained classes
    # unless we explicitly reject low-confidence guesses as "None" (a random,
    # untrained motion should not get confidently mapped onto a real gesture).
    if reject_threshold and top_confidence < reject_threshold:
        prediction = "None"
    else:
        prediction = top_k_list[0][0]

    history.append(prediction)
    uwb_plot_path = (
        save_uwb_plot(data, uwb_plot_dir, prediction, top_confidence)
        if uwb_plot_dir is not None
        else None
    )

    return {
        "prediction": prediction,
        "confidence": top_confidence,
        "top3": top_k_list,
        "history": list(history),
        "uwb_right_series": uwb_right_series,
        "uwb_left_series": uwb_left_series,
        "uwb_plot_file": str(uwb_plot_path) if uwb_plot_path else None,
    }


def make_handler(
    readers: dict[str, Any],
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
                        readers, model, feature_args, args.trial_seconds, args.top_k,
                        args.reject_threshold, history,
                        args.uwb_plot_dir if "uwb" in sensors else None,
                    )
                if result["uwb_plot_file"]:
                    print(f"Saved UWB plot: {result['uwb_plot_file']}")
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

    for reader in readers.values():
        reader.start()

    server = ThreadingHTTPServer(
        ("127.0.0.1", args.web_port),
        make_handler(readers, model, feature_args, args, sensors, accuracy, history, record_lock),
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
        # serve_forever() has already returned here.  shutdown() must be
        # invoked from a different thread, otherwise it can deadlock and
        # prevent the UWB reader from closing its FiRa session.
        server.server_close()
        for reader in readers.values():
            reader.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
