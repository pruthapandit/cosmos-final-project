# Gesture Recognition Final Project

Multi-sensor gesture recognition pipeline: IMU (wrist/palm), mmWave radar
(range profile), and UWB (two-way ranging between an anchor and two wrist
tags). Built around a coordinator-based collector so every sensor's samples
share one monotonic clock, cut into trials offline from event markers.

## Layout

- `collect.py` -- run a collection session across any combination of
  `imu`, `mmwave`, `uwb`.
- `sensors/` -- one reader per sensor (`imu_reader.py`, `mmwave_reader.py`,
  `uwb_reader.py`), all implementing the shared `BaseSensorReader` interface
  in `base_reader.py`.
- `get_range_profile.py` -- CLI handshake and TLV/frame parsing for the
  mmWave radar; `mmwave_reader.py` wraps this rather than duplicating it.
- `xwrL64xx-evm/*.cfg` -- radar CLI configs used by `--mmwave-cfg`.
- `combine_gesture_sessions.py` -- cuts accepted trials out of one or more
  `collect.py` sessions into a single dataset (`trials.csv` + per-trial
  `.npz` files), preserving `collector`/`gesture` labels for later
  leave-one-person-out evaluation.
- `test_gesture_fusion.py` -- quick leave-one-out comparison of IMU-only,
  mmWave-only, UWB-only, and all-three-fused accuracy on a combined dataset.

## Example

```bash
python collect.py --sensors imu mmwave uwb \
  --imu-port COM12 \
  --mmwave-port COM11 --mmwave-cfg xwrL64xx-evm/hand_distance.cfg \
  --uwb-controller-port COM13 --uwb-right-port COM15 --uwb-left-port COM14 \
  --uwb-channel 5 --uwb-preamble-idx 12 \
  --collector yourname --gestures Pull Push Right Left Clapping \
  --trials-per-gesture 6 --trial-seconds 2

python combine_gesture_sessions.py data/raw/session_* --output datasets/gesture_dataset_v1
python test_gesture_fusion.py datasets/gesture_dataset_v1
```

UWB ranging depends on a separate `uwb-qorvo-tools` checkout (Qorvo's FiRa
UCI tooling); point `--uwb-tools-root` at it. It also needs a Python
environment where `uci`/`fira` actually import cleanly (Python 3.13 breaks
a dynamic-enum patch this library relies on -- Python 3.9/3.10 works).
