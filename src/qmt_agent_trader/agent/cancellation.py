"""Thread-safe cooperative cancellation primitives for agent runs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Event, Lock

logger = logging.getLogger(__name__)


class CancellationToken:
    """A one-way cancellation signal safe to read from worker threads."""

    def __init__(self) -> None:
        self._event = Event()
        self._callbacks: set[Callable[[], None]] = set()
        self._callbacks_lock = Lock()

    def request_cancel(self) -> None:
        with self._callbacks_lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                # Cancellation must remain a one-way signal even when an
                # optional resource closer has its own cleanup failure.
                logger.exception("cancellation callback failed")

    def __call__(self) -> bool:
        """Allow the token to be passed anywhere a cancellation callback fits."""
        return self.is_cancel_requested()

    def add_cancel_callback(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register a resource closer and return an idempotent unregister hook."""
        invoke_now = False
        with self._callbacks_lock:
            if self._event.is_set():
                invoke_now = True
            else:
                self._callbacks.add(callback)

        if invoke_now:
            try:
                callback()
            except Exception:
                logger.exception("cancellation callback failed")

        removed = False

        def remove() -> None:
            nonlocal removed
            if removed:
                return
            removed = True
            with self._callbacks_lock:
                self._callbacks.discard(callback)

        return remove

    def is_cancel_requested(self) -> bool:
        return self._event.is_set()
