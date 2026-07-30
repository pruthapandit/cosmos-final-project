#!/usr/bin/env python3
"""Browser dashboard for predict_stream.py's continuous gesture prediction.

Same two-tier-confidence sliding-window logic as predict_stream.py, but
instead of printing to a terminal, it runs a small local HTTP server (no new
dependencies -- stdlib http.server only) and pushes live prediction updates,
plus a scrolling UWB right/left wrist-distance trace, to predict_stream_ui.html
over Server-Sent Events.

This is the experimental fork of predict_stream_ui.py: it adds the two-arm
false-positive guard (T-Arm / Raise Arms / Two-Arm Boxing must show real
movement on BOTH wrists' live UWB trace, not just the model's raw
confidence) plus --two-arm-debug/--two-arm-min-std/--two-arm-min-ratio for
tuning it against live data instead of only the offline training set.
predict_stream_ui.py stays on the simpler, already-working Clapping-only
threshold change as a stable fallback.
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
UWB_CHART_HISTORY = 100  # points kept for the UWB scrolling chart (~UWB_CHART_HISTORY * tick-seconds of history)

# Clapping's real-gesture confidence tends to sit lower than the other classes'
# even on clean trials, so it gets its own lower bar; everything else still
# uses --confidence-threshold.
CONFIDENCE_THRESHOLD_OVERRIDES = {"Clapping": 0.35}

# These are only ever genuine gestures when BOTH wrists move -- a one-arm
# gesture (or raising/punching with just one arm) occasionally scores high
# enough on one of these classes to slip past the confidence threshold.
# Requiring live UWB std on both sides (min_std_cm, min_ratio -- defaults
# below are tuned per gesture against training-data percentiles, but the
# live rolling window behaves differently from clean offline trial clips,
# so use --two-arm-debug to see the real numbers and --two-arm-min-std /
# --two-arm-min-ratio to override them without editing this file.
# min_ratio is min(right_std, left_std) / max(...). max_corr (only wired up
# for Two-Arm Boxing so far, None = skip) is the Pearson correlation between
# the two wrists' distance traces over the window; None disables the check.
#
# Two-Arm Boxing: a ratio-only gate can't work here -- live debug data showed
# a one-arm punch topping out at ratio=0.74, but requiring ratio>=0.75 also
# rejected most genuine two-arm boxing, and *any* ratio floor below 0.74
# lets that same one-arm leak back in. Replayed every labeled Two-Arm-Boxing
# vs One-Arm-Boxing trial (280 total, across all datasets/, same UWB
# out-of-range filter as train_gesture.py) through candidate thresholds:
# ratio>=0.68 alone recalls only 92% of genuine two-arm boxing at 28% one-arm
# leak. Two-arm boxing (alternating punches) shows strongly negative
# right/left correlation (median -0.51) that one-arm boxing mostly doesn't
# (median -0.04) -- adding "and corr<=-0.01" as a second, independent signal
# and relaxing the ratio floor to 0.60 recalls 98% at the *same* 27% leak
# rate: a strict improvement on both axes over ratio alone. min_std_cm
# barely moved either metric in this sweep (it's a "some real motion
# happened" floor, not a discriminator), so it's lowered rather than
# dropped. Still confirm with --two-arm-debug against live data -- the
# offline trial clips this was tuned on are cleaner than a live rolling
# window.
TWO_ARM_GESTURES: dict[str, tuple[float, float, float | None]] = {
    "T-Arm": (7.0, 0.70, None),
    "Raise Arms": (13.0, 0.75, None),
    "Two-Arm Boxing": (5.0, 0.60, -0.01),
}

# Gestures that are only genuine with a specific large, signed net movement
# on one wrist -- catches a class getting confidently misread off an
# unrelated motion (e.g. raising the right arm scoring as "Pull"). Values are
# (side, min_abs_delta_cm, sign): Pull reliably shows a +27..+62cm net
# right-wrist delta across every collector in training data (always
# positive, always the right wrist) -- nothing else comes close, so a
# smaller or wrong-signed delta is very unlikely to be a real Pull.
DIRECTIONAL_GESTURES: dict[str, tuple[str, float, int]] = {
    "Pull": ("right", 25.0, 1),
}


def _uwb_side_stds(data: dict[str, Any]) -> tuple[float, float]:
    """Live right/left UWB distance std (cm) over the current window, with the
    same out-of-range-reading filter train_gesture.py applies when fitting."""
    right = np.asarray(data.get("uwb_right_distance_cm", []), dtype=float)
    left = np.asarray(data.get("uwb_left_distance_cm", []), dtype=float)
    right = right[np.isfinite(right) & (right <= tg.UWB_MAX_PLAUSIBLE_CM)]
    left = left[np.isfinite(left) & (left <= tg.UWB_MAX_PLAUSIBLE_CM)]
    right_std = float(np.std(right)) if right.size > 1 else 0.0
    left_std = float(np.std(left)) if left.size > 1 else 0.0
    return right_std, left_std


def _uwb_side_corr(data: dict[str, Any]) -> float:
    """Pearson correlation between the right/left UWB distance traces over
    the current window (same out-of-range filter as _uwb_side_stds), applied
    only where both sides have a valid reading at the same tick. Two-arm
    gestures alternate (one wrist approaching while the other retreats),
    which shows up as a strong negative correlation; a one-arm gesture's
    near-stationary wrist doesn't track the moving one either way, so it
    stays close to zero. Returns 0.0 (no signal) if there aren't enough
    jointly-valid points."""
    right = np.asarray(data.get("uwb_right_distance_cm", []), dtype=float)
    left = np.asarray(data.get("uwb_left_distance_cm", []), dtype=float)
    n = min(right.size, left.size)
    if n < 3:
        return 0.0
    right, left = right[:n], left[:n]
    mask = (
        np.isfinite(right) & (right <= tg.UWB_MAX_PLAUSIBLE_CM)
        & np.isfinite(left) & (left <= tg.UWB_MAX_PLAUSIBLE_CM)
    )
    if mask.sum() < 3 or np.std(right[mask]) < 1e-9 or np.std(left[mask]) < 1e-9:
        return 0.0
    return float(np.corrcoef(right[mask], left[mask])[0, 1])


def _uwb_side_delta(data: dict[str, Any], side: str) -> float:
    """Net (end - start) UWB distance change (cm) for one wrist over the
    current window, with the same out-of-range-reading filter as above."""
    arr = np.asarray(data.get(f"uwb_{side}_distance_cm", []), dtype=float)
    arr = arr[np.isfinite(arr) & (arr <= tg.UWB_MAX_PLAUSIBLE_CM)]
    if arr.size < 2:
        return 0.0
    return float(arr[-1] - arr[0])


def _directional_check(
    prediction: str, data: dict[str, Any], sensors: Any
) -> tuple[bool, float] | None:
    """Returns (fails, delta), or None if this prediction has no directional
    guard (or there's no UWB data to check it with)."""
    spec = DIRECTIONAL_GESTURES.get(prediction)
    if spec is None or "uwb" not in sensors:
        return None
    side, min_abs_delta, sign = spec
    delta = _uwb_side_delta(data, side)
    fails = delta < min_abs_delta if sign > 0 else delta > -min_abs_delta
    return fails, delta


def _two_arm_thresholds(prediction: str, args: argparse.Namespace) -> tuple[float, float, float | None] | None:
    base = TWO_ARM_GESTURES.get(prediction)
    if base is None:
        return None
    min_std_cm = args.two_arm_min_std if args.two_arm_min_std is not None else base[0]
    min_ratio = args.two_arm_min_ratio if args.two_arm_min_ratio is not None else base[1]
    max_corr = args.two_arm_max_corr if args.two_arm_max_corr is not None else base[2]
    return min_std_cm, min_ratio, max_corr


def _two_arm_check(
    prediction: str, data: dict[str, Any], sensors: Any, args: argparse.Namespace
) -> tuple[bool, float, float, float, float | None] | None:
    """Returns (fails, right_std, left_std, ratio, corr), or None if this
    prediction has no two-arm guard (or there's no UWB data to check it
    with). corr is None when max_corr isn't configured for this gesture."""
    thresholds = _two_arm_thresholds(prediction, args)
    if thresholds is None or "uwb" not in sensors:
        return None
    min_std_cm, min_ratio, max_corr = thresholds
    right_std, left_std = _uwb_side_stds(data)
    lo, hi = sorted((right_std, left_std))
    ratio = lo / hi if hi > 1e-9 else 0.0
    fails = lo < min_std_cm or ratio < min_ratio
    corr = None
    if max_corr is not None:
        corr = _uwb_side_corr(data)
        fails = fails or corr > max_corr
    return fails, right_std, left_std, ratio, corr


class SharedState:
    def __init__(self, sensors: list[str], accuracy: float) -> None:
        self._lock = threading.Lock()
        self.sensors = sensors
        self.accuracy = accuracy
        self.prediction = "Idle"
        self.confidence = 0.0
        self.top3: list[tuple[str, float]] = []
        self.history: deque[str] = deque(maxlen=10)
        self.history.append("Idle")
        self.uwb_right_series: deque[float] = deque(maxlen=UWB_CHART_HISTORY)
        self.uwb_left_series: deque[float] = deque(maxlen=UWB_CHART_HISTORY)

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def push_gesture(self, gesture: str) -> None:
        with self._lock:
            self.history.append(gesture)

    def extend_uwb(self, right: list[float], left: list[float]) -> None:
        with self._lock:
            self.uwb_right_series.extend(right)
            self.uwb_left_series.extend(left)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sensors": self.sensors,
                "model_accuracy": self.accuracy,
                "prediction": self.prediction,
                "confidence": self.confidence,
                "top3": list(self.top3),
                "history": list(self.history),
                "uwb_right_series": [v for v in self.uwb_right_series if np.isfinite(v)],
                "uwb_left_series": [v for v in self.uwb_left_series if np.isfinite(v)],
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser dashboard for continuous gesture prediction.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--tick-seconds", type=float, default=0.1)
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--switch-to-gesture-votes", type=int, default=2)
    parser.add_argument("--switch-to-idle-votes", type=int, default=2)
    parser.add_argument("--web-port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab.")

    parser.add_argument(
        "--two-arm-min-std", type=float, default=None,
        help="Override min_std_cm for every two-arm gesture guard (T-Arm/Raise Arms/Two-Arm Boxing) "
             "instead of the per-gesture defaults, for live tuning.",
    )
    parser.add_argument(
        "--two-arm-min-ratio", type=float, default=None,
        help="Override min_ratio (smaller-side std / larger-side std) for every two-arm gesture guard.",
    )
    parser.add_argument(
        "--two-arm-max-corr", type=float, default=None,
        help="Override max_corr (right/left wrist-distance correlation ceiling) for every two-arm "
             "gesture guard that has one configured (currently just Two-Arm Boxing).",
    )
    parser.add_argument(
        "--two-arm-debug", action="store_true",
        help="Print live right/left UWB std + ratio + accept/reject verdict whenever a two-arm "
             "gesture clears the confidence threshold, so you can read off real numbers to tune with.",
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
            elif name == "mmwave":
                data.update(build_mmwave_dict(windowed, window_start))
            elif name == "uwb":
                data.update(build_uwb_dict(windowed, window_start))
                right = data.get("uwb_right_distance_cm", np.array([]))
                left = data.get("uwb_left_distance_cm", np.array([]))
                if right.size or left.size:
                    # Only the newest points each tick, so the chart scrolls
                    # instead of re-plotting the whole window every time.
                    take_r = right[-max(1, right.size // 5):] if right.size else np.array([])
                    take_l = left[-max(1, left.size // 5):] if left.size else np.array([])
                    state.extend_uwb(list(take_r.astype(float)), list(take_l.astype(float)))

        features, _ = tg.extract_features(data, feature_args)
        X = np.asarray([features], dtype=float)
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        top_idx = np.argsort(proba)[::-1][:3]
        top3 = [(str(classes[i]), float(proba[i])) for i in top_idx]
        raw_prediction = top3[0][0]
        top_confidence = top3[0][1]
        state.update(top3=top3)

        # Below its confidence threshold: not a transient blip to ignore, the
        # model doesn't confidently match ANY trained class -- report "None"
        # (same quick/safe persistence as Idle) instead of silently freezing
        # on whatever real gesture was last displayed.
        threshold = CONFIDENCE_THRESHOLD_OVERRIDES.get(raw_prediction, args.confidence_threshold)
        prediction = "None" if top_confidence < threshold else raw_prediction

        # Two-arm gestures need both wrists actually moving -- a one-arm
        # gesture that scored high on one of these classes gets bounced to
        # "None" here rather than displayed.
        if prediction in TWO_ARM_GESTURES:
            check = _two_arm_check(prediction, data, readers.keys(), args)
            if check is not None:
                fails, right_std, left_std, ratio, corr = check
                if args.two_arm_debug:
                    verdict = "REJECT" if fails else "accept"
                    corr_str = f"corr={corr:+.2f} " if corr is not None else ""
                    print(
                        f"[two-arm] {prediction:16s} right_std={right_std:5.1f}cm "
                        f"left_std={left_std:5.1f}cm ratio={ratio:.2f} {corr_str}-> {verdict}"
                    )
                if fails:
                    prediction = "None"

        # Directional gestures (currently just Pull) need a real, large,
        # correctly-signed net wrist movement -- an unrelated motion that
        # scored high on one of these classes gets bounced to "None" here.
        if prediction in DIRECTIONAL_GESTURES:
            dcheck = _directional_check(prediction, data, readers.keys())
            if dcheck is not None:
                dfails, delta = dcheck
                if args.two_arm_debug:
                    verdict = "REJECT" if dfails else "accept"
                    print(f"[directional] {prediction:16s} delta={delta:6.1f}cm -> {verdict}")
                if dfails:
                    prediction = "None"

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
        # serve_forever() has already returned here (including when Ctrl+C
        # interrupted it).  Calling shutdown() from that same thread can
        # deadlock, which previously prevented reader.stop() below from
        # sending the UWB controller's graceful session-stop command.
        server.server_close()
        for reader in readers.values():
            reader.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
