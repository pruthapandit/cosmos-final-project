#!/usr/bin/env python3
"""Train a gesture classifier from datasets built by combine_gesture_sessions.py.

Each trial is already one short, discrete gesture instance (not a continuous
recording needing windowing), so one trial's .npz == one training example.
Use --sensors to choose which modalities feed the model -- run it once with
a single sensor for a baseline, then again with all three for the fused
model, per the project's "compare single-sensor baseline vs fused" ask.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

STAT_NAMES = ("mean", "std", "min", "max", "p10", "p50", "p90", "start", "end", "delta", "slope")

# Derived from xwrL64xx-evm/hand_distance.cfg's chirpComnCfg/chirpTimingCfg (same
# formula as get_range_profile.py's parse_range_config): 128 raw bins spanning
# 0-5.86m. Bin 0 is a fixed antenna-coupling artifact (same magnitude regardless
# of gesture) and everything past ~2m is static room clutter -- both dwarf the
# actual hand-motion signal, which lives in the 0.15-2.0m band hand_lab.py
# already treats as the meaningful hand-gesture range.
MMWAVE_BIN_SPACING_M = 0.045744698791503904


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a gesture classifier from combine_gesture_sessions.py datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset folders created by combine_gesture_sessions.py.",
    )
    parser.add_argument(
        "--collectors",
        nargs="+",
        default=None,
        help="Only use trials recorded by these collectors. Defaults to every collector in the dataset.",
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        choices=["imu", "mmwave", "uwb"],
        default=["imu", "mmwave", "uwb"],
        help="Which sensors' features to include. Use one sensor for a single-sensor "
        "baseline, or all three (default) for the fused model.",
    )
    parser.add_argument("--imu-trajectory-points", type=int, default=10)
    parser.add_argument("--mmwave-time-bins", type=int, default=16)
    parser.add_argument("--mmwave-range-bins", type=int, default=24)
    parser.add_argument(
        "--mmwave-min-range-m", type=float, default=0.15,
        help="Crop the range profile to this near-field window before resampling/"
        "normalizing, so static clutter and the bin-0 antenna-coupling artifact "
        "outside the gesture-relevant range don't dominate the features.",
    )
    parser.add_argument("--mmwave-max-range-m", type=float, default=2.0)
    parser.add_argument("--mmwave-bin-spacing-m", type=float, default=MMWAVE_BIN_SPACING_M)
    parser.add_argument("--uwb-trajectory-points", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--augment-per-trial", type=int, default=0,
        help="Generate this many augmented copies of each TRAINING trial (raw-signal jitter/"
        "scale/rotation/time-warp for IMU, noise for mmWave/UWB) before feature extraction, to "
        "widen the motion-style variation the model sees. Never applied to the test split. 0 disables it.",
    )
    parser.add_argument(
        "--augment-intensity", type=float, default=1.0,
        help="Scales all augmentation perturbation magnitudes. Small/subtle gestures (Soli, Palm "
        "Up-Down, Making Fist and Open) have little signal to begin with, so too-strong "
        "augmentation can swamp them -- lower this if those classes degrade.",
    )
    parser.add_argument(
        "--augment-sensors", nargs="+", choices=["imu", "mmwave", "uwb"], default=None,
        help="Which sensors' raw signals to perturb when augmenting (defaults to --sensors). "
        "mmWave's per-trial self-normalization (scaled by that trial's own 95th percentile) makes "
        "already-low-signal classes unstable under added noise -- exclude mmwave here if it hurts "
        "while imu/uwb augmentation still helps.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--group-by",
        choices=["recording", "collector"],
        default="recording",
        help="Use 'collector' for a random cross-student split; see --test-collector "
        "for an explicit held-out-person evaluation.",
    )
    parser.add_argument(
        "--test-collector",
        default=None,
        help="Hold out this collector entirely for testing (everyone else trains). "
        "Overrides --group-by/--test-size when set.",
    )
    parser.add_argument(
        "--classifier",
        choices=["random_forest", "svm_rbf", "svm_poly", "decision_tree", "knn"],
        default="random_forest",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument(
        "--rf-max-depth", type=int, default=None,
        help="Cap RandomForest tree depth (sklearn default None = grow until pure/single-sample "
        "leaves, which can overfit with only ~10 trials/class/person). Try 8-15 to regularize.",
    )
    parser.add_argument(
        "--rf-min-samples-leaf", type=int, default=4,
        help="Minimum samples per leaf (sklearn default is 1, which lets a tree memorize a single "
        "training example -- risky with only ~10 trials/class/person). 4 measured as a clean win: "
        "+3 points held-out-person accuracy with zero cost to same-session accuracy.",
    )
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--svm-degree", type=int, default=3)
    parser.add_argument("--decision-tree-max-depth", type=int)
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--knn-weights", choices=["uniform", "distance"], default="distance")
    parser.add_argument("--model-out")
    parser.add_argument("--confusion-out")
    parser.add_argument(
        "--refit-full",
        action="store_true",
        help="After reporting held-out evaluation, refit the saved model on every usable trial. "
        "Use this for a deployment model so no collected data is left out.",
    )
    return parser.parse_args()


# ---- shared numeric helpers (self-contained -- no box_lab_common/posture_lab_common) ----


def fill_nan_series(values: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    if output.size == 0:
        return output
    finite = np.isfinite(output)
    if not finite.any():
        output[:] = fallback
        return output
    if finite.all():
        return output
    indices = np.arange(output.size)
    output[~finite] = np.interp(indices[~finite], indices[finite], output[finite])
    return output


def resample_vector(values: np.ndarray, target_count: int) -> np.ndarray:
    values = fill_nan_series(np.asarray(values, dtype=float))
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if values.size == 0:
        return np.zeros(target_count, dtype=float)
    if values.size == 1:
        return np.full(target_count, float(values[0]), dtype=float)
    source_x = np.linspace(0.0, 1.0, values.size)
    target_x = np.linspace(0.0, 1.0, target_count)
    return np.interp(target_x, source_x, values)


def resample_matrix(matrix: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.zeros((target_rows, target_cols), dtype=float)
    time_resampled = np.column_stack(
        [resample_vector(matrix[:, col], target_rows) for col in range(matrix.shape[1])]
    )
    if matrix.shape[1] == 1:
        return np.repeat(time_resampled, target_cols, axis=1)
    source_x = np.linspace(0.0, 1.0, matrix.shape[1])
    target_x = np.linspace(0.0, 1.0, target_cols)
    return np.vstack([np.interp(target_x, source_x, row) for row in time_resampled])


def robust_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    scale = float(np.percentile(np.abs(finite), 95))
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = float(np.std(finite))
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return np.clip(values / scale, -6.0, 6.0)


OSCILLATION_STAT_NAMES = ("peak_count", "peak_rate_hz")


def oscillation_stats(values: np.ndarray, time_s: np.ndarray) -> list[float]:
    """Count threshold up-crossings, to tell "one big sweep" gestures (Left,
    T-Arm: one smooth hump) apart from repetitive ones (Making Fist and
    Open, Soli: many small spikes) -- summary stats and a coarse resampled
    trajectory both wash out that difference since they only capture
    overall level/shape, not how many times the signal oscillates.
    """
    values = fill_nan_series(np.asarray(values, dtype=float))
    if values.size < 3:
        return [0.0, 0.0]
    if time_s.size == values.size and values.size > 1:
        duration = max(float(time_s[-1] - time_s[0]), 1e-9)
    else:
        duration = 1.0
    span = float(np.max(values) - np.min(values))
    if span <= 1e-9:
        return [0.0, 0.0]
    threshold = float(np.min(values)) + 0.5 * span
    above = values > threshold
    crossings = int(np.sum((above[1:].astype(int) - above[:-1].astype(int)) == 1))
    return [float(crossings), float(crossings / duration)]


def series_stats(values: np.ndarray, time_s: np.ndarray) -> list[float]:
    values = fill_nan_series(np.asarray(values, dtype=float))
    if values.size == 0:
        return [0.0] * len(STAT_NAMES)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        values = np.zeros_like(values, dtype=float)
        finite = values
    if time_s.size != values.size:
        time_s = np.linspace(0.0, values.size - 1, values.size)
    duration = max(float(time_s[-1] - time_s[0]), 1e-9) if values.size > 1 else 1.0
    slope = float((values[-1] - values[0]) / duration)
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
        float(np.percentile(finite, 10)),
        float(np.percentile(finite, 50)),
        float(np.percentile(finite, 90)),
        float(values[0]),
        float(values[-1]),
        float(values[-1] - values[0]),
        slope,
    ]


# ---- per-sensor feature extraction ----

IMU_AXES = ("ax", "ay", "az", "gx", "gy", "gz")


def extract_imu_features(data: dict[str, np.ndarray], trajectory_points: int) -> tuple[list[float], list[str]]:
    time_s = np.asarray(data.get("imu_time_s", []), dtype=float)
    series = {axis: np.asarray(data.get(f"imu_{axis}", []), dtype=float) for axis in IMU_AXES}

    if all(series[a].size == series["ax"].size for a in IMU_AXES) and series["ax"].size:
        accel_mag = np.sqrt(series["ax"] ** 2 + series["ay"] ** 2 + series["az"] ** 2)
        gyro_mag = np.sqrt(series["gx"] ** 2 + series["gy"] ** 2 + series["gz"] ** 2)
    else:
        accel_mag = np.empty(0, dtype=float)
        gyro_mag = np.empty(0, dtype=float)

    features: list[float] = []
    names: list[str] = []
    for axis_name, arr in list(series.items()) + [("accel_mag", accel_mag), ("gyro_mag", gyro_mag)]:
        features.extend(series_stats(arr, time_s))
        names.extend(f"imu_{axis_name}_{stat}" for stat in STAT_NAMES)
    for axis_name, arr in (("accel_mag", accel_mag), ("gyro_mag", gyro_mag)):
        trajectory = resample_vector(arr, trajectory_points)
        features.extend(float(v) for v in trajectory)
        names.extend(f"imu_{axis_name}_t{i:02d}" for i in range(trajectory_points))
    for axis_name, arr in (("accel_mag", accel_mag), ("gyro_mag", gyro_mag)):
        features.extend(oscillation_stats(arr, time_s))
        names.extend(f"imu_{axis_name}_{stat}" for stat in OSCILLATION_STAT_NAMES)
    return features, names


def crop_mmwave_range(profile: np.ndarray, min_range_m: float, max_range_m: float, bin_spacing_m: float) -> np.ndarray:
    if profile.ndim != 2 or profile.size == 0:
        return profile
    num_bins = profile.shape[1]
    start_bin = max(0, int(math.floor(min_range_m / bin_spacing_m)))
    stop_bin = min(num_bins, int(math.ceil(max_range_m / bin_spacing_m)) + 1)
    if start_bin >= stop_bin:
        return profile
    return profile[:, start_bin:stop_bin]


def extract_mmwave_features(
    data: dict[str, np.ndarray],
    time_bins: int,
    range_bins: int,
    min_range_m: float = 0.15,
    max_range_m: float = 2.0,
    bin_spacing_m: float = MMWAVE_BIN_SPACING_M,
) -> tuple[list[float], list[str]]:
    profile = np.asarray(data.get("mmwave_range_profile", np.empty((0, 0))), dtype=float)
    profile = crop_mmwave_range(profile, min_range_m, max_range_m, bin_spacing_m)
    if profile.ndim != 2 or profile.size == 0:
        image = np.zeros((time_bins, range_bins), dtype=float)
    else:
        image = robust_normalize(resample_matrix(profile, time_bins, range_bins))
    features = [float(v) for v in image.ravel()]
    names = [f"mmwave_img_t{t:02d}_b{b:02d}" for t in range(time_bins) for b in range(range_bins)]
    return features, names


UWB_MAX_PLAUSIBLE_CM = 500.0  # the anchor/wrist setup never exceeds ~2m; anything
# past this is the board's raw 16-bit "invalid reading" sentinel (65535) slipping
# through with a status that still reads "Ok" -- a single one of these can drag an
# entire trial's mean/std/max features toward thousands of cm.


def extract_uwb_features(data: dict[str, np.ndarray], trajectory_points: int) -> tuple[list[float], list[str]]:
    time_s = np.asarray(data.get("uwb_time_s", []), dtype=float)
    features: list[float] = []
    names: list[str] = []
    for label in ("right", "left"):
        arr = np.asarray(data.get(f"uwb_{label}_distance_cm", []), dtype=float)
        arr = np.where(arr > UWB_MAX_PLAUSIBLE_CM, np.nan, arr)
        features.extend(series_stats(arr, time_s))
        names.extend(f"uwb_{label}_{stat}" for stat in STAT_NAMES)
        trajectory = resample_vector(arr, trajectory_points)
        features.extend(float(v) for v in trajectory)
        names.extend(f"uwb_{label}_t{i:02d}" for i in range(trajectory_points))
    return features, names


def extract_features(data: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[list[float], list[str]]:
    features: list[float] = []
    names: list[str] = []
    if "imu" in args.sensors:
        f, n = extract_imu_features(data, args.imu_trajectory_points)
        features.extend(f)
        names.extend(n)
    if "mmwave" in args.sensors:
        f, n = extract_mmwave_features(
            data,
            args.mmwave_time_bins,
            args.mmwave_range_bins,
            args.mmwave_min_range_m,
            args.mmwave_max_range_m,
            args.mmwave_bin_spacing_m,
        )
        features.extend(f)
        names.extend(n)
    if "uwb" in args.sensors:
        f, n = extract_uwb_features(data, args.uwb_trajectory_points)
        features.extend(f)
        names.extend(n)
    return features, names


# ---- training-set-only data augmentation ----
# Applied to raw per-trial signals (before feature extraction), on the
# training split only -- never the test split, so evaluation stays honest.
# Meant to widen the person-to-person motion-style variation the model sees,
# since each collector only contributed ~10 real trials per class.


def _small_rotation_matrix(rng: np.random.Generator, max_degrees: float) -> np.ndarray:
    """A small random 3D rotation, simulating the IMU being mounted at a
    slightly different wrist orientation than in training."""
    axis = rng.normal(size=3)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    angle = np.deg2rad(rng.uniform(-max_degrees, max_degrees))
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ])


def _time_warp(values: np.ndarray, warp: float) -> np.ndarray:
    """Speeds up or slows down the signal slightly (simulates performing the
    gesture a bit faster/slower) while keeping the same sample count. Takes
    a precomputed `warp` factor (rather than sampling its own) so every axis
    of a multi-channel signal can share the same warp and stay time-synced --
    warping each channel independently would desynchronize axes that are
    supposed to move together, destroying cross-axis shape/timing info."""
    n = values.size
    if n < 3:
        return values
    x = np.linspace(0.0, 1.0, n)
    warped_x = np.clip(x**warp, 0.0, 1.0)
    return np.interp(x, warped_x, values)


def augment_imu(data: dict[str, np.ndarray], rng: np.random.Generator, intensity: float = 1.0) -> dict[str, np.ndarray]:
    out = dict(data)
    axes = [np.asarray(data.get(f"imu_{a}", []), dtype=float) for a in IMU_AXES]
    if not all(a.size == axes[0].size and a.size > 0 for a in axes):
        return out

    accel = np.stack(axes[:3], axis=1)  # (n, 3): ax, ay, az
    gyro = np.stack(axes[3:], axis=1)   # (n, 3): gx, gy, gz

    # Same small rotation applied to both accel and gyro -- a real mounting-
    # angle change would affect both identically.
    rotation = _small_rotation_matrix(rng, max_degrees=3.0 * intensity)
    accel = accel @ rotation.T
    gyro = gyro @ rotation.T

    accel_scale = rng.uniform(1.0 - 0.03 * intensity, 1.0 + 0.03 * intensity)
    gyro_scale = rng.uniform(1.0 - 0.03 * intensity, 1.0 + 0.03 * intensity)
    accel = accel * accel_scale
    gyro = gyro * gyro_scale

    accel = accel + rng.normal(scale=0.01 * intensity, size=accel.shape)
    gyro = gyro + rng.normal(scale=0.5 * intensity, size=gyro.shape)

    # One shared warp for every axis so accel/gyro stay time-synced with
    # each other (see _time_warp's docstring for why per-axis warp is wrong).
    warp = 1.0 + rng.uniform(-0.04 * intensity, 0.04 * intensity)
    for i, name in enumerate(("ax", "ay", "az")):
        out[f"imu_{name}"] = _time_warp(accel[:, i], warp)
    for i, name in enumerate(("gx", "gy", "gz")):
        out[f"imu_{name}"] = _time_warp(gyro[:, i], warp)
    return out


def augment_mmwave(data: dict[str, np.ndarray], rng: np.random.Generator, intensity: float = 1.0) -> dict[str, np.ndarray]:
    out = dict(data)
    profile = np.asarray(data.get("mmwave_range_profile", np.empty((0, 0))), dtype=float)
    if profile.ndim != 2 or profile.size == 0:
        return out
    noisy = profile * (1.0 + rng.normal(scale=0.015 * intensity, size=profile.shape))
    noisy = noisy + rng.normal(scale=profile.std() * 0.01 * intensity + 1e-6, size=profile.shape)
    out["mmwave_range_profile"] = np.clip(noisy, 0, None)
    return out


def augment_uwb(data: dict[str, np.ndarray], rng: np.random.Generator, intensity: float = 1.0) -> dict[str, np.ndarray]:
    out = dict(data)
    # Shared warp across both wrists -- they're one physical trial timeline.
    warp = 1.0 + rng.uniform(-0.04 * intensity, 0.04 * intensity)
    for label in ("right", "left"):
        key = f"uwb_{label}_distance_cm"
        arr = np.asarray(data.get(key, []), dtype=float)
        if arr.size == 0:
            continue
        noisy = arr + rng.normal(scale=0.7 * intensity, size=arr.shape)
        out[key] = _time_warp(noisy, warp)
    return out


def augment_raw_data(
    data: dict[str, np.ndarray], rng: np.random.Generator, sensors: list[str], intensity: float = 1.0
) -> dict[str, np.ndarray]:
    out = dict(data)
    if "imu" in sensors:
        out.update(augment_imu(out, rng, intensity))
    if "mmwave" in sensors:
        out.update(augment_mmwave(out, rng, intensity))
    if "uwb" in sensors:
        out.update(augment_uwb(out, rng, intensity))
    return out


# ---- dataset loading ----


def read_rows(dataset_dir: Path) -> list[dict[str, str]]:
    manifest = dataset_dir / "trials.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"missing trials.csv: {manifest}")
    with manifest.open(newline="") as file:
        return list(csv.DictReader(file))


def build_examples(dataset_dirs: list[Path], args: argparse.Namespace):
    examples: list[list[float]] = []
    labels: list[str] = []
    recording_groups: list[str] = []
    collector_groups: list[str] = []
    raw_data: list[dict[str, np.ndarray]] = []
    feature_names: list[str] | None = None
    skipped: list[dict[str, str]] = []
    collectors: set[str] = set()

    for dataset_dir in dataset_dirs:
        for row in read_rows(dataset_dir):
            if args.collectors is not None and row["collector"] not in args.collectors:
                continue
            npz_path = Path(row["npz_path"])
            if not npz_path.exists():
                skipped.append({"npz_path": str(npz_path), "reason": "missing npz"})
                continue
            with np.load(npz_path) as npz:
                data = {key: npz[key] for key in npz.files}

            features, names = extract_features(data, args)
            if feature_names is None:
                feature_names = names
            elif len(names) != len(feature_names):
                skipped.append({"npz_path": str(npz_path), "reason": "feature length mismatch"})
                continue

            examples.append(features)
            labels.append(row["gesture"])
            recording_groups.append(str(npz_path))
            collector_groups.append(row["collector"])
            collectors.add(row["collector"])
            raw_data.append(data)

    return (
        examples,
        labels,
        recording_groups,
        collector_groups,
        feature_names or [],
        skipped,
        collectors,
        raw_data,
    )


# ---- classifier + split (mirrors train_posture.py's conventions) ----


def parse_svm_gamma(value: str) -> str | float:
    if value in {"scale", "auto"}:
        return value
    try:
        gamma = float(value)
    except ValueError as exc:
        raise SystemExit("--svm-gamma must be scale, auto, or a positive float.") from exc
    if gamma <= 0:
        raise SystemExit("--svm-gamma must be positive.")
    return gamma


def classifier_label(classifier: str) -> str:
    return {
        "random_forest": "Random Forest",
        "svm_rbf": "RBF SVM",
        "svm_poly": "Polynomial SVM",
        "decision_tree": "Decision Tree",
        "knn": "KNN",
    }[classifier]


def build_classifier(args: argparse.Namespace, train_count: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    if args.classifier == "random_forest":
        params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.rf_max_depth,
            "min_samples_leaf": args.rf_min_samples_leaf,
            "random_state": args.random_state,
            "class_weight": "balanced",
        }
        return RandomForestClassifier(**params), params

    if args.classifier == "svm_rbf":
        params = {
            "kernel": "rbf",
            "C": args.svm_c,
            "gamma": parse_svm_gamma(args.svm_gamma),
            "class_weight": "balanced",
            "probability": True,
            "random_state": args.random_state,
        }
        return make_pipeline(StandardScaler(), SVC(**params)), params

    if args.classifier == "svm_poly":
        params = {
            "kernel": "poly",
            "degree": args.svm_degree,
            "C": args.svm_c,
            "gamma": parse_svm_gamma(args.svm_gamma),
            "class_weight": "balanced",
            "probability": True,
            "random_state": args.random_state,
        }
        return make_pipeline(StandardScaler(), SVC(**params)), params

    if args.classifier == "decision_tree":
        params = {
            "max_depth": args.decision_tree_max_depth,
            "random_state": args.random_state,
            "class_weight": "balanced",
        }
        return DecisionTreeClassifier(**params), params

    if args.classifier == "knn":
        requested_neighbors = max(1, int(args.knn_neighbors))
        actual_neighbors = min(requested_neighbors, int(train_count))
        params = {"n_neighbors": actual_neighbors, "weights": args.knn_weights}
        return make_pipeline(StandardScaler(), KNeighborsClassifier(**params)), params

    raise SystemExit(f"Unsupported classifier: {args.classifier}")


def group_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray, args: argparse.Namespace):
    from sklearn.model_selection import GroupShuffleSplit

    unique_groups = sorted(set(groups))
    unique_labels = sorted(set(y))
    if len(unique_groups) < 2:
        return None

    for offset in range(80):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=args.test_size,
            random_state=args.random_state + offset,
        )
        train_idx, test_idx = next(splitter.split(X, y, groups))
        if set(y[train_idx]) == set(unique_labels) and set(y[test_idx]) == set(unique_labels):
            return train_idx, test_idx
    return None


def split_examples(
    X: np.ndarray,
    y: np.ndarray,
    recording_groups: np.ndarray,
    collector_groups: np.ndarray,
    args: argparse.Namespace,
):
    from sklearn.model_selection import train_test_split

    if args.test_collector:
        test_mask = collector_groups == args.test_collector
        if not test_mask.any():
            raise SystemExit(
                f"--test-collector {args.test_collector!r} not found among "
                f"collectors: {sorted(set(collector_groups))}"
            )
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        return train_idx, test_idx, f"held_out_collector:{args.test_collector}"

    if args.group_by == "collector":
        collector_split = group_split(X, y, collector_groups, args)
        if collector_split is not None:
            return collector_split[0], collector_split[1], "collector"

    recording_split = group_split(X, y, recording_groups, args)
    if recording_split is not None:
        mode = "recording" if args.group_by == "recording" else "recording_fallback"
        return recording_split[0], recording_split[1], mode

    indices = np.arange(len(y))
    class_counts = {label: int((y == label).sum()) for label in sorted(set(y))}
    stratify = y if min(class_counts.values()) >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )
    return train_idx, test_idx, "segment"


def default_model_path(dataset_dirs: list[Path], classifier: str, sensors: list[str]) -> Path:
    sensor_tag = "_".join(sensors)
    name = f"{classifier}_{sensor_tag}_gesture_{timestamp()}.joblib"
    if len(dataset_dirs) == 1:
        return dataset_dirs[0] / "models" / name
    return Path("models") / name


def main() -> int:
    args = parse_args()
    if not 0.0 < args.test_size < 1.0:
        raise SystemExit("--test-size must be between 0 and 1.")

    try:
        import joblib
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            accuracy_score,
            classification_report,
            confusion_matrix,
        )
    except ImportError as exc:
        raise SystemExit(f"Training dependencies missing: {exc}") from exc

    dataset_dirs = [Path(item).expanduser().resolve() for item in args.datasets]
    (
        examples,
        labels,
        recording_groups,
        collector_groups,
        feature_names,
        skipped,
        collectors,
        raw_data,
    ) = build_examples(dataset_dirs, args)

    if len(set(labels)) < 2:
        raise SystemExit("Need at least two gesture labels to train.")
    if len(examples) < 4:
        raise SystemExit("Need at least four usable trials to train.")

    X = np.asarray(examples, dtype=float)
    y = np.asarray(labels)
    recording_group_array = np.asarray(recording_groups)
    collector_group_array = np.asarray(collector_groups)
    class_counts = {label: int((y == label).sum()) for label in sorted(set(labels))}

    train_idx, test_idx, split_mode = split_examples(
        X, y, recording_group_array, collector_group_array, args
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    if len(set(y_train)) < 2:
        raise SystemExit("Training split needs at least two gesture labels.")

    if args.augment_per_trial > 0:
        rng = np.random.default_rng(args.random_state)
        augment_sensors = args.augment_sensors if args.augment_sensors is not None else args.sensors
        augmented_X: list[list[float]] = []
        augmented_y: list[str] = []
        for idx in train_idx:
            for _ in range(args.augment_per_trial):
                augmented_raw = augment_raw_data(raw_data[idx], rng, augment_sensors, args.augment_intensity)
                features, _ = extract_features(augmented_raw, args)
                augmented_X.append(features)
                augmented_y.append(labels[idx])
        X_train = np.vstack([X_train, np.asarray(augmented_X, dtype=float)])
        y_train = np.concatenate([y_train, np.asarray(augmented_y)])
        print(f"Augmented training set: {len(train_idx)} real trials -> {len(X_train)} "
              f"(+{len(augmented_X)} synthetic, {args.augment_per_trial}x per trial)")

    missing_from_train = sorted(set(y_test) - set(y_train))
    if missing_from_train:
        print(
            f"Warning: test set contains gestures never seen in training: {missing_from_train} "
            "-- the model cannot possibly predict these correctly."
        )

    model, classifier_params = build_classifier(args, train_count=len(X_train))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    labels_order = sorted(set(y))
    matrix = confusion_matrix(y_test, predictions, labels=labels_order)
    report_text = classification_report(y_test, predictions, labels=labels_order, zero_division=0)
    report_dict = classification_report(
        y_test, predictions, labels=labels_order, output_dict=True, zero_division=0
    )

    # Keep the held-out metrics above honest, then use the complete labelled
    # dataset for the model that will actually be deployed live.
    if args.refit_full:
        model.fit(X, y)

    model_out = (
        Path(args.model_out).expanduser().resolve()
        if args.model_out
        else default_model_path(dataset_dirs, args.classifier, args.sensors)
    )
    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "sensors": args.sensors,
        "feature_names": feature_names,
        "imu_trajectory_points": args.imu_trajectory_points,
        "mmwave_time_bins": args.mmwave_time_bins,
        "mmwave_range_bins": args.mmwave_range_bins,
        "mmwave_min_range_m": args.mmwave_min_range_m,
        "mmwave_max_range_m": args.mmwave_max_range_m,
        "mmwave_bin_spacing_m": args.mmwave_bin_spacing_m,
        "uwb_trajectory_points": args.uwb_trajectory_points,
        "classifier": args.classifier,
        "classifier_label": classifier_label(args.classifier),
        "classifier_params": classifier_params,
        "group_by": args.group_by,
        "test_collector": args.test_collector,
        "split_mode": split_mode,
        "class_counts": class_counts,
        "labels_order": labels_order,
        "collectors": sorted(collectors),
        "collector_filter": args.collectors,
        "accuracy": accuracy,
        "refit_full": args.refit_full,
        "fitted_trial_count": int(len(y) if args.refit_full else len(X_train)),
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "datasets": [str(path) for path in dataset_dirs],
        "skipped": skipped,
    }
    joblib.dump(payload, model_out)

    confusion_out = (
        Path(args.confusion_out).expanduser().resolve()
        if args.confusion_out
        else model_out.with_suffix(".confusion.png")
    )
    confusion_out.parent.mkdir(parents=True, exist_ok=True)
    # Default figure size doesn't scale with class count, so long/rotated
    # labels (e.g. "Making Fist and Open") collide once there are ~15 of
    # them; size the figure to the label count and anchor the rotated
    # labels at their own edge so they line up cleanly instead of overlapping.
    n_classes = len(labels_order)
    fig_side = max(8.0, n_classes * 0.62)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side * 0.82))
    display = ConfusionMatrixDisplay(matrix, display_labels=labels_order)
    display.plot(cmap="Blues", values_format="d", xticks_rotation=45, ax=ax, colorbar=True)
    ax.set_title(f"Gesture Classifier [{'+'.join(args.sensors)}] ({accuracy:.2%} accuracy)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(confusion_out, dpi=180)
    plt.close()

    print(f"Sensors: {'+'.join(args.sensors)}")
    print(f"Collectors: {', '.join(sorted(collectors))}")
    print(f"Usable trials: {len(examples)}")
    print(f"Class counts: {class_counts}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Split mode: {split_mode}; train={len(X_train)}, test={len(X_test)}")
    if args.refit_full:
        print(f"Refit saved model on all {len(y)} usable trials after held-out evaluation.")
    if split_mode == "segment":
        print("Warning: split fell back to a plain stratified split (group split was too small).")
    if split_mode == "recording_fallback":
        print("Warning: collector split was not possible; used recording split instead.")
    if skipped:
        print(f"Skipped items: {len(skipped)}")
        for item in skipped[:10]:
            print(f"  - {item.get('npz_path', '')}: {item['reason']}")
    print(report_text)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Saved model: {model_out}")
    print(f"Saved confusion matrix: {confusion_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
