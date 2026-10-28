"""Hourly cap on chat-model calls, so a traffic spike cannot run up the bill.

    if not model_budget.try_use():
        return BUSY_ANSWER
"""

import os
import threading
import time

DEFAULT_CALLS_PER_HOUR = 200  # override with the MODEL_CALLS_PER_HOUR environment variable

# Fixed reply when the cap is reached (no model call, so no cost).
BUSY_ANSWER = (
    "I am handling many questions right now. Please try again in a few minutes, "
    "or check pagasa.dost.gov.ph and your local DRRMO.\n"
    "Maraming tanong ang sinasagot ko ngayon. Subukan muli sa ilang minuto, "
    "o tingnan ang pagasa.dost.gov.ph at ang inyong lokal na DRRMO."
)


class CallBudget:
    """Allows at most ``max_calls`` uses in any rolling window of ``window_seconds``."""

    def __init__(self, max_calls: int, window_seconds: float = 3600):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._call_times: list[float] = []
        self._lock = threading.Lock()

    def try_use(self) -> bool:
        """Record one call and return True, or return False if the cap is reached."""
        now = time.monotonic()
        with self._lock:
            cutoff = now - self.window_seconds
            self._call_times = [t for t in self._call_times if t > cutoff]
            if len(self._call_times) >= self.max_calls:
                return False
            self._call_times.append(now)
            return True


model_budget = CallBudget(int(os.getenv("MODEL_CALLS_PER_HOUR", DEFAULT_CALLS_PER_HOUR)))
