from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any


@dataclass
class SessionItem:
    created_at: datetime
    expires_at: datetime
    data: dict[str, Any]


class SplitSessionStore:
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._ttl = ttl_seconds
        self._items: dict[str, SessionItem] = {}
        self._lock = Lock()

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def cleanup_expired(self) -> int:
        now = self._now()
        removed = 0
        with self._lock:
            dead = [k for k, v in self._items.items() if v.expires_at <= now]
            for k in dead:
                self._items.pop(k, None)
                removed += 1
        return removed

    def set(self, session_id: str, data: dict[str, Any]) -> None:
        now = self._now()
        item = SessionItem(
            created_at=now,
            expires_at=now + timedelta(seconds=self._ttl),
            data=data,
        )
        with self._lock:
            self._items[session_id] = item

    def get(self, session_id: str) -> dict[str, Any] | None:
        self.cleanup_expired()
        with self._lock:
            item = self._items.get(session_id)
            if not item:
                return None
            return item.data

