"""Sliding-Window Rate Limiter for SaaS Tenant Traffic Control."""
import time
import threading
from typing import Dict, List, Tuple

class SlidingWindowRateLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._history: Dict[str, List[float]] = {}

    def is_allowed(self, key: str, max_requests_per_minute: int) -> Tuple[bool, int, float]:
        """Checks if request is allowed under sliding 60-second window.
        Returns: (is_allowed, current_requests_in_window, retry_after_seconds)
        """
        now = time.time()
        window_start = now - 60.0

        with self._lock:
            if key not in self._history:
                self._history[key] = []
            
            # Prune timestamps older than 60 seconds
            timestamps = [t for t in self._history[key] if t > window_start]
            self._history[key] = timestamps

            count = len(timestamps)
            if count >= max_requests_per_minute:
                oldest = timestamps[0]
                retry_after = round(max(0.1, 60.0 - (now - oldest)), 1)
                return False, count, retry_after

            # Record this request
            self._history[key].append(now)
            return True, count + 1, 0.0

    def reset(self, key: str = None):
        with self._lock:
            if key:
                self._history.pop(key, None)
            else:
                self._history.clear()

# Global rate limiter instance
rate_limiter = SlidingWindowRateLimiter()
