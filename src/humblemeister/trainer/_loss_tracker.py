from __future__ import annotations

import math
from collections import deque


class LossBreakthroughDetector:
    """
    Tracks a rolling window of recent loss values and detects statistically
    significant drops that warrant an immediate checkpoint.

    Each call to update() consumes a new loss value and returns True if the
    value is a breakthrough — defined as dropping more than `threshold` times
    the mean quadratic deviation (standard deviation) below the window mean.

    The detector is inactive (always returns False) until the window is full.

    Args:
        window:    number of past loss values to track (default 16)
        threshold: number of standard deviations below the mean that
                   constitutes a breakthrough (default 3.0)
    """

    def __init__(self, window: int = 16, threshold: float = 3.0) -> None:
        self._window = window
        self._threshold = threshold
        self._history: deque[float] = deque(maxlen=window)

    def update(self, loss: float) -> bool:
        """
        Consume a new loss value.

        Returns True if the value represents a significant breakthrough
        relative to the recent history, False otherwise.
        The new value is added to the history after the check.
        """
        trigger = False

        if len(self._history) == self._window:
            mean = sum(self._history) / self._window
            mqd = math.sqrt(sum((x - mean) ** 2 for x in self._history) / self._window)
            trigger = mqd > 0.0 and loss < mean - self._threshold * mqd

        self._history.append(loss)
        return trigger

    @property
    def ready(self) -> bool:
        """True once the window is full and the detector is active."""
        return len(self._history) == self._window
