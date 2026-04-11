from __future__ import annotations

import math
from collections import deque


class LossBreakthroughDetector:
    """
    Detects sustained loss breakthroughs by comparing a recent window against
    an older baseline window.

    Maintains two separate deques:
        - old_window   (default 16 points): baseline — tracks mean and MQD
        - recent_window (default 4 points): current trend — tracks mean

    A breakthrough is triggered when the recent mean drops more than
    `threshold` MQDs below the old mean, indicating a durable shift rather
    than a transient spike.

    The detector is inactive until both windows are full (20 points consumed).

    Args:
        old_window:    number of baseline loss values (default 16)
        recent_window: number of recent loss values (default 4)
        threshold:     MQD multiples below old mean that constitute a
                       breakthrough (default 3.0)
    """

    def __init__(
        self,
        old_window: int = 16,
        recent_window: int = 4,
        threshold: float = 3.0,
    ) -> None:
        self._old_window = old_window
        self._recent_window = recent_window
        self._threshold = threshold
        self._old: deque[float] = deque(maxlen=old_window)
        self._recent: deque[float] = deque(maxlen=recent_window)

    def update(self, loss: float) -> bool:
        """
        Consume a new loss value.

        Returns True if the recent window mean has dropped more than
        `threshold` * old_MQD below the old window mean, indicating a
        sustained breakthrough. The new value is added after the check.
        """
        trigger = False

        if self.ready:
            old_mean = sum(self._old) / self._old_window
            old_mqd = math.sqrt(sum((x - old_mean) ** 2 for x in self._old) / self._old_window)
            recent_mean = sum(self._recent) / self._recent_window
            trigger = old_mqd > 0.0 and recent_mean < old_mean - self._threshold * old_mqd

        # oldest recent value graduates into the old window
        if len(self._recent) == self._recent_window:
            self._old.append(self._recent[0])
        self._recent.append(loss)

        return trigger

    @property
    def ready(self) -> bool:
        """True once both windows are full and the detector is active."""
        return len(self._old) == self._old_window and len(self._recent) == self._recent_window
