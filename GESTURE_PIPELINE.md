# Gesture Recognition Final Project -- Command Reference

Full command reference for the collect -> combine -> train pipeline. See
`sensors/` for the reader implementations and `collect.py`/
`combine_gesture_sessions.py`/`train_gesture.py` for the actual code.

Ports below (`COM11`/`COM12`/`COM13`/`COM14`/`COM15`) match this project's
current wiring -- update them if a board gets replugged into a different
port. Run everything from the `py39` conda environment; the UWB tooling
(`uci`/`fira`) breaks under Python 3.13.

## 1. Collect a session

```powershell
conda activate py39
python collect.py --sensors imu mmwave uwb `
  --imu-port COM12 `
  --mmwave-port COM11 --mmwave-cfg xwrL64xx-evm/hand_distance.cfg `
  --uwb-controller-port COM15 --uwb-right-port COM13 --uwb-left-port COM14 `
  --collector <name> --trials-per-gesture 10
```

- Omit `--gestures` to use the full 15-gesture official list (defined in
  `GESTURES` at the top of `collect.py`). Pass a subset (e.g.
  `--gestures Pull Push Right Left Clapping`) to collect only specific
  gestures -- useful for backfilling a tester who's missing a few.
- `--no-shuffle` collects in fixed gesture-block order instead of the
  default randomized order (easier for the tester to follow, at the cost
  of a session-time confound the shuffle avoids).
- `--trial-seconds 2.0` is the default; bump it (`--trial-seconds 3.0`) for
  slower/more complex gestures if the tester feels rushed.
- `--imu-rate-hz 25` (default) throttles IMU samples pushed downstream;
  pass `0` to disable throttling and get the full ~100Hz stream.
- `--dry-run` runs trials and prints sample counts as normal but writes
  nothing to disk -- use this for testing the pipeline/hardware without
  polluting `data/raw/`.
- `Ctrl+C` mid-session stops gracefully: all sensors (including the UWB
  subprocesses) are shut down cleanly and whatever trials completed so far
  are already saved (every trial is flushed to disk immediately, not
  buffered until the session ends).
- Each session writes to `data/raw/session_<timestamp>/` --
  `session_metadata.json`, `events.csv`, `trials.csv`, and one CSV per
  sensor (`imu.csv`, `mmwave.csv`, `uwb.csv`).

Only sensors you actually have wired need to be listed in `--sensors`; the
corresponding `--<sensor>-port` args are required only for sensors you include.

## 2. Combine sessions into a dataset

```powershell
python combine_gesture_sessions.py data/raw/session_A data/raw/session_B ... --output datasets/<dataset_name>
```

- Accepts any number of raw session directories -- combine multiple
  sessions from the same collector (e.g. a main session + a backfill for
  missing gestures) or multiple collectors at once.
- Slices each sensor's CSV to every accepted trial's real time window
  (rejected/repeated trials are dropped automatically) and writes one
  `trials.csv` manifest plus per-trial `.npz` payloads to
  `datasets/<dataset_name>/`.
- Prints a JSON summary: per-collector trial counts, per-gesture trial
  counts, and any skipped trials -- check this after every combine to
  confirm the counts match what you expect.

## 3. Train and evaluate

```powershell
python train_gesture.py datasets/<dataset_name> --sensors imu mmwave uwb
```

- Default split is a random 75/25 split over trials (grouped by
  recording). Prints overall accuracy, a full per-class
  precision/recall/F1 report, and saves a `.joblib` model plus a
  `.confusion.png` confusion matrix under `datasets/<dataset_name>/models/`.

**Single-sensor vs. fused comparison** (baseline vs. improved model):

```powershell
python train_gesture.py datasets/<dataset_name> --sensors imu
python train_gesture.py datasets/<dataset_name> --sensors mmwave
python train_gesture.py datasets/<dataset_name> --sensors uwb
python train_gesture.py datasets/<dataset_name> --sensors imu mmwave uwb
```

**Held-out-person (across-user) evaluation:**

```powershell
python train_gesture.py datasets/<dataset_name> --sensors imu mmwave uwb --test-collector <name> --model-out datasets/<dataset_name>/models/heldout_<name>.joblib
```

Trains on everyone except `<name>`, tests only on `<name>`'s trials. Run
once per collector to get a full leave-one-person-out picture.

**Classifier comparison** (second "two models" axis for the rubric):

```powershell
python train_gesture.py datasets/<dataset_name> --sensors imu mmwave uwb --classifier random_forest
python train_gesture.py datasets/<dataset_name> --sensors imu mmwave uwb --classifier svm
python train_gesture.py datasets/<dataset_name> --sensors imu mmwave uwb --classifier knn
python train_gesture.py datasets/<dataset_name> --sensors imu mmwave uwb --classifier decision_tree
```

`--group-by collector` splits by collector instead of by recording (a
weaker, but still-useful, generalization check when you don't want to name
a specific held-out person).

## Typical end-to-end example

```powershell
conda activate py39

# Collect (repeat per collector)
python collect.py --sensors imu mmwave uwb --imu-port COM12 --mmwave-port COM11 --mmwave-cfg xwrL64xx-evm/hand_distance.cfg --uwb-controller-port COM15 --uwb-right-port COM13 --uwb-left-port COM14 --collector eric --trials-per-gesture 10

# Combine everyone's sessions
python combine_gesture_sessions.py data/raw/session_<eric1> data/raw/session_<eric2> data/raw/session_<shanmu1> data/raw/session_<shanmu2> --output datasets/gesture_final_v1

# Baseline vs fused
python train_gesture.py datasets/gesture_final_v1 --sensors imu
python train_gesture.py datasets/gesture_final_v1 --sensors imu mmwave uwb

# Held-out-person
python train_gesture.py datasets/gesture_final_v1 --sensors imu mmwave uwb --test-collector eric --model-out datasets/gesture_final_v1/models/heldout_eric.joblib
python train_gesture.py datasets/gesture_final_v1 --sensors imu mmwave uwb --test-collector shanmu --model-out datasets/gesture_final_v1/models/heldout_shanmu.joblib
```

## Troubleshooting

- **UWB `RangingRxTimeout` spam / a board's blue light goes out**: usually
  means the board's session got stuck; unplug/replug that board and
  restart `collect.py`. Ctrl+C now shuts down gracefully so this shouldn't
  leave a stale session active, but if you do see the reader print
  `ignored N rounds from a stale/foreign UWB session handle`, that's the
  reader auto-filtering a leftover session rather than a live problem.
- **Two UWB boards on the same USB hub**: don't -- a shared hub's single
  Transaction Translator can't reliably serialize two simultaneously-active
  UWB boards. Plug all three boards directly into the laptop.
- **`AttributeError` from the `uci`/`fira` library**: you're running under
  Python 3.13. Use the `py39` conda environment for anything UWB-related.
- **Sample count printed live during a trial looks huge or tiny**: this is
  normal -- the printed count now reflects only the trial's real time
  window, not idle time between trials.
