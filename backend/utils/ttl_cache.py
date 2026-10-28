"""Small in-memory cache whose entries expire after a fixed time.

Two ways to use it:

    cache = TTLCache(ttl_seconds=60)
    cache.set("key", "value")
    cache.get("key")            # "value", or MISSING once expired

    @ttl_cache(ttl_seconds=60)  # caches a function's results by its arguments
    def slow_call(city): ...
"""

import functools
import threading
import time

# Returned by ``TTLCache.get`` when there is no fresh entry. A separate marker
# is used so that ``None`` and empty strings can still be cached.
MISSING = object()


class TTLCache:
    """Key/value store where each entry expires ``ttl_seconds`` after it was set."""

    def __init__(self, ttl_seconds: float):
        self.ttl_seconds = ttl_seconds
        self._entries: dict = {}  # key -> (expires_at, value)
        self._lock = threading.Lock()

    def get(self, key):
        """Return the cached value, or ``MISSING`` if absent or expired."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return MISSING
            expires_at, value = entry
            if expires_at < time.monotonic():
                del self._entries[key]
                return MISSING
            return value

    def set(self, key, value) -> None:
        """Store a value; it expires ``ttl_seconds`` from now."""
        with self._lock:
            self._entries[key] = (time.monotonic() + self.ttl_seconds, value)

    def clear(self) -> None:
        """Remove every entry."""
        with self._lock:
            self._entries.clear()


def ttl_cache(ttl_seconds: float):
    """Decorator that caches a function's results for ``ttl_seconds``.

    Arguments must be hashable. Calls that raise are not cached.
    """

    def decorator(function):
        cache = TTLCache(ttl_seconds)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            cached = cache.get(key)
            if cached is not MISSING:
                return cached
            result = function(*args, **kwargs)
            cache.set(key, result)
            return result

        wrapper.cache = cache  # lets tests clear it
        return wrapper

    return decorator
