"""Common interface for all sensor readers used in the final project pipeline.

Every reader collects raw device samples on a background thread and exposes
them through one small, sensor-agnostic API so collect.py (the coordinator)
never needs to know about serial ports, TLV parsing, or subprocess pipes for
any particular sensor.

Timestamps:
    Every sample is tagged with time.monotonic() at the moment it is
    received by *this process*, not the sensor's own onboard clock. Using
    one shared monotonic clock across every reader is what lets the
    coordinator line up samples from different sensors against a single
    trial_start / trial_end event stream later.
"""

from __future__ import annotations

import abc
import queue
import threading
import time
from typing import Any


class BaseSensorReader(abc.ABC):
    """Abstract base class every sensor reader implements.

    Subclasses implement _open, _close, and _run. Everything else
    (threading, queueing, draining) is handled here so each sensor-specific
    reader only has to deal with its own parsing logic.
    """

    name: str = "sensor"

    def __init__(self) -> None:
        self._queue: "queue.Queue[tuple[float, dict[str, Any]]]" = queue.Queue(maxsize=5000)
        self._errors: "queue.Queue[str]" = queue.Queue(maxsize=200)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- lifecycle ----------------------------------------------------
    def start(self) -> None:
        """Open the device and start the background reader thread."""
        self._stop_event.clear()
        try:
            self._open()
            self._thread = threading.Thread(target=self._run, daemon=True, name=f"{self.name}-reader")
            self._thread.start()
        except BaseException:
            # Startup can be interrupted after a device/subprocess has been
            # opened (notably during the UWB controlee startup delay).  Do
            # not leave that partial session running or its serial port held.
            self._stop_event.set()
            self._close()
            self._thread = None
            raise

    def stop(self) -> None:
        """Stop the reader thread and release the device."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._close()

    # ---- consumption ----------------------------------------------------
    def drain(self) -> tuple[list[tuple[float, dict[str, Any]]], list[str]]:
        """Return and clear every sample/error queued since the last call.

        Call this from the coordinator's main loop only -- never from
        inside a reader thread.
        """
        samples: list[tuple[float, dict[str, Any]]] = []
        while True:
            try:
                samples.append(self._queue.get_nowait())
            except queue.Empty:
                break

        errors: list[str] = []
        while True:
            try:
                errors.append(self._errors.get_nowait())
            except queue.Empty:
                break

        return samples, errors

    # ---- subclass hooks ----------------------------------------------------
    @abc.abstractmethod
    def _open(self) -> None:
        """Open the serial port / subprocess / socket. Called by start()."""

    @abc.abstractmethod
    def _close(self) -> None:
        """Release the device. Called by stop()."""

    @abc.abstractmethod
    def _run(self) -> None:
        """Background loop: read raw data, parse it, push samples to the queue.

        Must check self._stop_event regularly and return promptly once it
        is set. Must never let an exception escape this method -- catch it
        and call self._push_error() instead, so one sensor failing doesn't
        take down the others.
        """

    # ---- helpers for subclasses ----------------------------------------------------
    def _push_sample(self, fields: dict[str, Any]) -> None:
        """Subclasses call this once per parsed sample. Adds the shared clock."""
        item = (time.monotonic(), fields)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Drop the oldest sample rather than block the reader thread.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(item)

    def _push_error(self, message: str) -> None:
        try:
            self._errors.put_nowait(message)
        except queue.Full:
            pass
