#!/usr/bin/env python3
"""Quick check: does fusing IMU + mmwave + UWB beat any sensor alone?

Meant for a first, small batch of trials -- not the final evaluation. Loads
trials from one or more datasets built by combine_gesture_sessions.py,
extracts a small stats feature vector per sensor, and compares leave-one-out
accuracy for each sensor alone against every sensor concatenated (early
fusion). A dataset missing a sensor entirely (e.g. no uwb.csv collected yet)
still works -- that sensor's features just default to zeros.

With only a handful of trials per gesture, leave-one-out is used instead of
a train/test split so every trial gets to be the held-out example once.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

IMU_AXES = ("ax", "ay", "az", "gx", "gy", "gz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare IMU-only, mmwave-only, and fused leave-one-out accuracy."
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset folders created by combine_gesture_sessions.py.",
    )
    parser.add_argument("--bins", type=int, default=32, help="Resampled mmwave range-profile bin count.")
    parser.add_argument("--classifier", choices=["knn", "random_forest"], default="knn")
    parser.add_argument("--knn-neighbors", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def resample_vector(values: np.ndarray, target_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros(target_count, dtype=float)
    if values.size == 1:
        return np.full(target_count, float(values[0]), dtype=float)
    source_x = np.linspace(0.0, 1.0, values.size)
    target_x = np.linspace(0.0, 1.0, target_count)
    return np.interp(target_x, source_x, values)


def imu_features(data: dict[str, np.ndarray]) -> tuple[list[float], list[str]]:
    values: list[float] = []
    names: list[str] = []
    for axis in IMU_AXES:
        arr = np.asarray(data.get(f"imu_{axis}", []), dtype=float)
        if arr.size == 0:
            values.extend([0.0, 0.0])
        else:
            values.extend([float(np.mean(arr)), float(np.std(arr))])
        names.extend([f"imu_{axis}_mean", f"imu_{axis}_std"])
    return values, names


def mmwave_features(data: dict[str, np.ndarray], bins: int) -> tuple[list[float], list[str]]:
    profile = np.asarray(data.get("mmwave_range_profile", np.empty((0, 0))), dtype=float)
    if profile.ndim != 2 or profile.size == 0:
        mean_vec = np.zeros(bins, dtype=float)
        std_vec = np.zeros(bins, dtype=float)
    else:
        mean_vec = resample_vector(np.mean(profile, axis=0), bins)
        std_vec = resample_vector(np.std(profile, axis=0), bins)
    values = list(mean_vec) + list(std_vec)
    names = [f"mmwave_mean_b{i:02d}" for i in range(bins)] + [f"mmwave_std_b{i:02d}" for i in range(bins)]
    return values, names


def uwb_features(data: dict[str, np.ndarray]) -> tuple[list[float], list[str]]:
    values: list[float] = []
    names: list[str] = []
    for label in ("right", "left"):
        arr = np.asarray(data.get(f"uwb_{label}_distance_cm", []), dtype=float)
        finite = arr[np.isfinite(arr)] if arr.size else arr
        if finite.size == 0:
            values.extend([0.0, 0.0])
        else:
            values.extend([float(np.mean(finite)), float(np.std(finite))])
        names.extend([f"uwb_{label}_distance_mean_cm", f"uwb_{label}_distance_std_cm"])
    return values, names


def read_rows(dataset_dir: Path) -> list[dict[str, str]]:
    manifest = dataset_dir / "trials.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"missing trials.csv: {manifest}")
    with manifest.open(newline="") as file:
        return list(csv.DictReader(file))


def build_dataset(dataset_dirs: list[Path], bins: int):
    imu_rows: list[list[float]] = []
    mmwave_rows: list[list[float]] = []
    uwb_rows: list[list[float]] = []
    labels: list[str] = []
    collectors: list[str] = []
    skipped: list[dict[str, str]] = []

    for dataset_dir in dataset_dirs:
        rows = read_rows(dataset_dir)
        for row in rows:
            npz_path = Path(row["npz_path"])
            if not npz_path.exists():
                skipped.append({"npz_path": str(npz_path), "reason": "missing npz"})
                continue
            with np.load(npz_path) as npz:
                data = {key: npz[key] for key in npz.files}

            imu_vec, _imu_names = imu_features(data)
            mmwave_vec, _mmwave_names = mmwave_features(data, bins)
            uwb_vec, _uwb_names = uwb_features(data)

            imu_rows.append(imu_vec)
            mmwave_rows.append(mmwave_vec)
            uwb_rows.append(uwb_vec)
            labels.append(row["gesture"])
            collectors.append(row["collector"])

    return (
        np.array(imu_rows, dtype=float),
        np.array(mmwave_rows, dtype=float),
        np.array(uwb_rows, dtype=float),
        np.array(labels),
        np.array(collectors),
        skipped,
    )


def build_classifier(args: argparse.Namespace, train_count: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if args.classifier == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            class_weight="balanced",
        )

    neighbors = max(1, min(args.knn_neighbors, train_count))
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=neighbors))


def leave_one_out_predictions(X: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    from sklearn.model_selection import LeaveOneOut

    predictions = np.empty(len(y), dtype=y.dtype)
    for train_index, test_index in LeaveOneOut().split(X):
        classifier = build_classifier(args, len(train_index))
        classifier.fit(X[train_index], y[train_index])
        predictions[test_index] = classifier.predict(X[test_index])
    return predictions


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{name}: {accuracy * 100:.1f}% leave-one-out accuracy ({len(y_true)} trials)")
    return accuracy


def confusion_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    from sklearn.metrics import confusion_matrix

    labels = sorted(set(y_true))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    width = max(len(label) for label in labels) + 2
    header = " " * width + "".join(f"{label[:8]:>10}" for label in labels)
    print(header)
    for label, row in zip(labels, matrix):
        print(f"{label:<{width}}" + "".join(f"{value:>10d}" for value in row))


def main() -> int:
    args = parse_args()
    dataset_dirs = [Path(item).expanduser().resolve() for item in args.datasets]

    X_imu, X_mmwave, X_uwb, y, collectors, skipped = build_dataset(dataset_dirs, args.bins)

    if skipped:
        print(f"Skipped {len(skipped)} row(s) with missing npz files.")

    if len(y) < 3:
        print("Not enough trials to evaluate (need at least 3).")
        return 1

    gesture_counts = {str(label): int(np.sum(y == label)) for label in sorted(set(y))}
    print(f"Trials: {len(y)}  Gestures: {gesture_counts}  Collectors: {sorted(str(c) for c in set(collectors))}")

    fused_name = "Fused (IMU+mmwave+UWB)"
    X_fused = np.concatenate([X_imu, X_mmwave, X_uwb], axis=1)

    single_sensor_names = ["IMU-only", "mmwave-only", "UWB-only"]
    results = {}
    for name, X in zip(single_sensor_names + [fused_name], [X_imu, X_mmwave, X_uwb, X_fused]):
        predictions = leave_one_out_predictions(X, y, args)
        results[name] = report(name, y, predictions)
        if name == fused_name:
            print("Confusion matrix (fused model):")
            confusion_report(y, predictions)

    print("\nSummary:")
    for name, accuracy in results.items():
        print(f"  {name:<22} {accuracy * 100:5.1f}%")

    best_single = max(single_sensor_names, key=lambda name: results[name])
    if results[fused_name] > results[best_single]:
        print("\nFusion improved accuracy over every single sensor.")
    else:
        print(f"\nFusion did not beat the best single sensor ({best_single}) on this small sample --"
              " worth collecting more trials before drawing conclusions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
