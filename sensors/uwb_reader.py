"""UWB sensor reader: two wrist tags ranging against one fixed anchor.

Wraps Qorvo's uwb-qorvo-tools FiRa one-to-many two-way-ranging (TWR) demo
(run_fira_twr.py) behind the BaseSensorReader interface. Three physical
boards are involved -- one controller (anchor, fixed next to the mmWave
radar) and two controlees (right-wrist and left-wrist tags) -- each a
separate subprocess talking to its own serial port, per the one-to-many
example in uwb-qorvo-tools' run_fira_twr README:

    controller: --node onetomany --dest-mac [0x01,0x02] --n_controlees 2
    controlee 1 (right wrist): --node onetomany --controlee --mac 1
    controlee 2 (left wrist):  --node onetomany --controlee --mac 2

The controller's own stdout reports one distance measurement per wrist per
ranging round (tagged by mac address), so only its output needs to be
parsed live; the two controlee subprocesses just need to be running.

Each subprocess is launched with `-t -1` ("range forever, stop on
keypress") and a stdin pipe, so stopping the reader means writing a
newline to each process's stdin -- letting the Qorvo tool run its own
graceful ranging_stop/session_deinit sequence, the same way you would
press Enter in a manual terminal. This avoids leaving a board stuck in an
active ranging session, which is what previously required the
reset_calibration/load_cal/reset_device recovery sequence.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from sensors.base_reader import BaseSensorReader


class OneToManyRangeParser:
    """Parses run_fira_twr.py controller stdout into one row per ranging round.

    Each "sequence n:" line starts a new round; each "# Measurement N:"
    sub-block reports one controlee's status/mac/distance. Rounds are
    finalized (and returned) when the next "sequence n:" line arrives, so
    call flush() once after the stream ends to get the last round.

    The controller's stdout can contain more than one "session handle" --
    in practice this happens when a prior run left a session active on the
    board that never got torn down, and it keeps reporting its own (always
    failing) rounds alongside the real one. Only the first session handle
    seen is treated as real; rounds from any other handle are parsed (so
    they don't corrupt the real round's in-progress fields) but discarded
    instead of being emitted as samples.
    """

    sequence_re = re.compile(r"sequence n:\s*(\d+)")
    session_handle_re = re.compile(r"session handle:\s*(\d+)")
    measurement_re = re.compile(r"#\s*Measurement\s+\d+:")
    status_re = re.compile(r"status:\s*([A-Za-z0-9_]+)\s*\((0x[0-9a-fA-F]+)\)")
    mac_re = re.compile(r"mac address:\s*([0-9A-Fa-f.:]+)")
    distance_re = re.compile(r"distance:\s*([-+]?[0-9]*\.?[0-9]+)\s*cm")

    def __init__(self) -> None:
        self.sequence: int | None = None
        self._current_mac: str | None = None
        self._current_status: str | None = None
        self.pending: dict[str, Any] = {}
        self._current_session_handle: int | None = None
        self._target_session_handle: int | None = None
        self._ignoring_current_round = False
        self.discarded_foreign_rounds = 0

    @staticmethod
    def _label_for_mac(mac_text: str | None) -> str | None:
        """Map a printed mac address (format varies: '01.00', '00:01', ...) to a wrist label."""
        if not mac_text:
            return None
        values = [int(token, 16) for token in re.findall(r"[0-9A-Fa-f]+", mac_text)]
        has_right = 1 in values
        has_left = 2 in values
        if has_right and not has_left:
            return "right"
        if has_left and not has_right:
            return "left"
        return None

    def _finalize(self) -> dict[str, Any]:
        sample = {
            "sequence": self.pending.get("sequence", -1),
            "right_distance_cm": self.pending.get("right_distance_cm", ""),
            "right_status": self.pending.get("right_status", "missing"),
            "left_distance_cm": self.pending.get("left_distance_cm", ""),
            "left_status": self.pending.get("left_status", "missing"),
        }
        self.pending = {}
        return sample

    def feed(self, line: str) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []

        session_match = self.session_handle_re.search(line)
        if session_match:
            self._current_session_handle = int(session_match.group(1))
            return samples

        sequence_match = self.sequence_re.search(line)
        if sequence_match:
            if self.pending:
                samples.append(self._finalize())
            if self._target_session_handle is None:
                self._target_session_handle = self._current_session_handle
            if self._current_session_handle == self._target_session_handle:
                self._ignoring_current_round = False
                self.sequence = int(sequence_match.group(1))
                self.pending = {"sequence": self.sequence}
            else:
                # A different session handle is reporting rounds -- almost
                # certainly a stale session left active from a previous run.
                # Ignore every field until the next round starts, so it can't
                # leak into (or get emitted as) real data.
                self._ignoring_current_round = True
                self.discarded_foreign_rounds += 1
                self.sequence = None
                self.pending = {}
            self._current_mac = None
            self._current_status = None
            return samples

        if self._ignoring_current_round:
            return samples

        if self.measurement_re.search(line):
            self._current_mac = None
            self._current_status = None
            return samples

        status_match = self.status_re.search(line)
        if status_match:
            self._current_status = status_match.group(1)
            return samples

        mac_match = self.mac_re.search(line)
        if mac_match:
            self._current_mac = mac_match.group(1)
            return samples

        distance_match = self.distance_re.search(line)
        if distance_match:
            label = self._label_for_mac(self._current_mac)
            if label:
                self.pending[f"{label}_distance_cm"] = float(distance_match.group(1))
                self.pending[f"{label}_status"] = self._current_status or "unknown"
            return samples

        return samples

    def flush(self) -> dict[str, Any] | None:
        if not self.pending:
            return None
        return self._finalize()


class UwbReader(BaseSensorReader):
    """Ranges two wrist-worn UWB tags against one fixed anchor (FiRa one-to-many TWR)."""

    name = "uwb"

    def __init__(
        self,
        controller_port: str,
        right_port: str,
        left_port: str,
        uwb_tools_root: Path,
        channel: int,
        preamble_code: int,
        session_id: int = 42,
        slot_span: int = 2400,
        slots_per_rr: int = 25,
        ranging_span: int = 50,
        startup_delay: float = 5.0,
        stop_timeout: float = 5.0,
        python_exe: str = sys.executable,
        log_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.controller_port = controller_port
        self.right_port = right_port
        self.left_port = left_port
        self.uwb_tools_root = Path(uwb_tools_root)
        self.channel = channel
        self.preamble_code = preamble_code
        self.session_id = session_id
        self.slot_span = slot_span
        self.slots_per_rr = slots_per_rr
        self.ranging_span = ranging_span
        self.startup_delay = startup_delay
        self.stop_timeout = stop_timeout
        self.python_exe = python_exe
        self.log_dir = Path(log_dir) if log_dir else None

        self._run_fira_twr = (
            self.uwb_tools_root / "scripts" / "fira" / "run_fira_twr" / "run_fira_twr.py"
        )
        self._right_proc: subprocess.Popen | None = None
        self._left_proc: subprocess.Popen | None = None
        self._controller_proc: subprocess.Popen | None = None
        self._right_log: Any = None
        self._left_log: Any = None
        self._controller_log: Any = None

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        parts = [
            str(self.uwb_tools_root),
            str(self.uwb_tools_root / "lib" / "uwb-uci"),
            str(self.uwb_tools_root / "lib" / "uqt-utils"),
        ]
        existing = env.get("PYTHONPATH")
        if existing:
            parts.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(parts)
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _shared_flags(self) -> list[str]:
        return [
            "--channel", str(self.channel),
            "--preamble-idx", str(self.preamble_code),
            "--session", str(self.session_id),
            "--slot-span", str(self.slot_span),
            "--slots-per-rr", str(self.slots_per_rr),
            "--ranging-span", str(self.ranging_span),
            "--aoa-report", "all-disabled",
            "--node", "onetomany",
            "-t", "-1",
        ]

    def _controlee_command(self, port: str, mac: int) -> list[str]:
        return [
            self.python_exe, "-u", str(self._run_fira_twr),
            "-p", port,
            "--controlee",
            "--mac", str(mac),
            *self._shared_flags(),
        ]

    def _controller_command(self) -> list[str]:
        return [
            self.python_exe, "-u", str(self._run_fira_twr),
            "-p", self.controller_port,
            "--dest-mac", "[0x01,0x02]",
            "--n_controlees", "2",
            "--stats",
            *self._shared_flags(),
        ]

    def _spawn(self, cmd: list[str], log_path: Path | None, capture_stdout: bool) -> tuple[subprocess.Popen, Any]:
        log_file: Any = subprocess.DEVNULL
        if log_path is not None:
            log_file = open(log_path, "w", buffering=1)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE if capture_stdout else log_file,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self._env(),
        )
        return proc, log_file

    def _open(self) -> None:
        if not self._run_fira_twr.exists():
            raise FileNotFoundError(
                f"run_fira_twr.py not found under uwb_tools_root: {self._run_fira_twr}"
            )

        # Always keep raw logs somewhere -- without them, a stuck/crashed
        # controller subprocess looks identical to "just no rounds yet", with
        # no way to tell the difference after the fact.
        if self.log_dir is None:
            self.log_dir = Path("uwb_reader_logs") / time.strftime("%Y%m%d_%H%M%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[uwb] raw subprocess logs: {self.log_dir}")

        right_log_path = self.log_dir / "uwb_right_controlee_log.txt"
        left_log_path = self.log_dir / "uwb_left_controlee_log.txt"
        controller_log_path = self.log_dir / "uwb_controller_log.txt"

        self._right_proc, self._right_log = self._spawn(
            self._controlee_command(self.right_port, mac=1), right_log_path, capture_stdout=False
        )
        self._left_proc, self._left_log = self._spawn(
            self._controlee_command(self.left_port, mac=2), left_log_path, capture_stdout=False
        )

        time.sleep(self.startup_delay)

        self._controller_proc, _ = self._spawn(
            self._controller_command(), None, capture_stdout=True
        )
        self._controller_log = open(controller_log_path, "w", buffering=1)

    def _graceful_stop(self, proc: subprocess.Popen | None) -> None:
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin:
                proc.stdin.write("\n")
                proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            proc.wait(timeout=self.stop_timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _close(self) -> None:
        # Stop the controller first so it finishes ranging_stop/session_deinit
        # and prints its final stats before the controlees disappear.
        self._graceful_stop(self._controller_proc)
        self._graceful_stop(self._right_proc)
        self._graceful_stop(self._left_proc)

        if self._right_log not in (None, subprocess.DEVNULL):
            self._right_log.close()
        if self._left_log not in (None, subprocess.DEVNULL):
            self._left_log.close()
        if self._controller_log is not None:
            self._controller_log.close()

    _FAILURE_MARKERS = (
        "critical", "traceback", "error", "timeout", "rejected", "failed", "exception",
    )

    def _run(self) -> None:
        assert self._controller_proc is not None and self._controller_proc.stdout is not None
        parser = OneToManyRangeParser()
        saw_ranging_data = False
        try:
            for line in self._controller_proc.stdout:
                if self._controller_log is not None:
                    self._controller_log.write(line)
                if self._stop_event.is_set():
                    break

                samples = parser.feed(line)

                lowered = line.lower()
                if "ranging data" in lowered:
                    saw_ranging_data = True
                elif "slot in error" in lowered:
                    pass  # a normal per-measurement field name, not a failure signal
                elif parser._ignoring_current_round:
                    pass  # noise from a stale/foreign session handle, not a real failure
                elif any(marker in lowered for marker in self._FAILURE_MARKERS):
                    self._push_error(f"uwb controller: {line.strip()}")

                for sample in samples:
                    self._push_sample(sample)
        except Exception as exc:  # keep one sensor's failure from taking down the others
            self._push_error(f"uwb: reader loop failed: {exc}")
            return

        final_sample = parser.flush()
        if final_sample is not None:
            self._push_sample(final_sample)

        if parser.discarded_foreign_rounds:
            print(
                f"[uwb] ignored {parser.discarded_foreign_rounds} rounds from a stale/foreign "
                "UWB session handle (most likely left active by a previous run) -- these "
                "are not in uwb.csv and were not counted as real timeouts"
            )

        if not saw_ranging_data:
            self._push_error(
                "uwb: controller never printed any ranging data -- check "
                f"{self.log_dir / 'uwb_controller_log.txt'} and the controlee logs "
                "in the same folder for the real error."
            )


if __name__ == "__main__":
    # Quick manual smoke test:
    #   python -m sensors.uwb_reader --controller-port COM15 --right-port COM13
    #     --left-port COM14 --uwb-tools-root "C:\Users\...\UWB_lab\uwb-qorvo-tools"
    #     --channel 5 --preamble-idx 12
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the UWB reader alone.")
    parser.add_argument("--controller-port", required=True)
    parser.add_argument("--right-port", required=True)
    parser.add_argument("--left-port", required=True)
    parser.add_argument("--uwb-tools-root", required=True, type=Path)
    parser.add_argument("--channel", type=int, required=True)
    parser.add_argument("--preamble-idx", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=15.0)
    args = parser.parse_args()

    reader = UwbReader(
        controller_port=args.controller_port,
        right_port=args.right_port,
        left_port=args.left_port,
        uwb_tools_root=args.uwb_tools_root,
        channel=args.channel,
        preamble_code=args.preamble_idx,
    )
    reader.start()
    print(f"Ranging for {args.seconds:g}s... (Ctrl+C to stop early)")
    start = time.monotonic()
    total = 0
    interrupted = False
    try:
        while time.monotonic() - start < args.seconds:
            samples, errors = reader.drain()
            for err in errors:
                print(f"[uwb] {err}")
            total += len(samples)
            for ts, fields in samples:
                print(f"[uwb] t={ts:.3f} {fields}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[uwb] Interrupted -- stopping subprocesses gracefully...")
    finally:
        reader.stop()
        time.sleep(0.3)  # let a final diagnostic (e.g. "never produced data") land
        samples, errors = reader.drain()
        for err in errors:
            print(f"[uwb] {err}")
        total += len(samples)
    elapsed = time.monotonic() - start
    status = "Stopped early" if interrupted else "Done"
    print(f"{status}. Received {total} rounds in {elapsed:.1f}s.")
