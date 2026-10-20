"""进程内短期缓存：上传文件 → upload_id，inspect 状态 → token。

单用户本机应用，进程重启后这些临时映射丢失是合理的（文件仍在 uploads/）。
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Optional
from uuid import uuid4


class TTLCache:
    def __init__(self, ttl: float = 6 * 3600, max_items: int = 200):
        self.ttl = ttl
        self.max_items = max_items
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, value: Any, key: Optional[str] = None) -> str:
        key = key or uuid4().hex
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)
        return key

    def get(self, key: str) -> Optional[Any]:
        if not key:
            return None
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            ts, value = item
            if time.time() - ts > self.ttl:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def drop(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


uploads = TTLCache(ttl=24 * 3600, max_items=100)
inspections = TTLCache(ttl=6 * 3600, max_items=100)
