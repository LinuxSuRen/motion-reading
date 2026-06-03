import time
from typing import Dict, Tuple


class JointSmoother:
    def __init__(
        self,
        alpha: float = 0.3,
        threshold: float = 0.02,
        min_interval: float = 0.05,
    ):
        self.alpha = alpha
        self.threshold = threshold
        self.min_interval = min_interval
        self._smoothed: Dict[str, float] = {}
        self._last_sent: Dict[str, float] = {}
        self._last_send_time: float = 0.0

    def filter(self, joints: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        now = time.time()

        if now - self._last_send_time < self.min_interval:
            return False, {}

        smoothed = {}
        for k, v in joints.items():
            if k in self._smoothed:
                smoothed[k] = self.alpha * v + (1 - self.alpha) * self._smoothed[k]
            else:
                smoothed[k] = v

        self._smoothed = smoothed

        if self._last_sent:
            max_change = max(
                abs(smoothed.get(k, 0) - self._last_sent.get(k, 0))
                for k in smoothed
            )
            if max_change < self.threshold:
                return False, {}

        return True, smoothed

    def mark_sent(self, joints: Dict[str, float]):
        self._last_sent = dict(joints)
        self._last_send_time = time.time()

    def reset(self):
        self._smoothed.clear()
        self._last_sent.clear()
        self._last_send_time = 0.0
