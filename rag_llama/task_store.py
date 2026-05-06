from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any


class TaskStore:
    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _load(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save(self, tasks: list[dict[str, Any]]) -> None:
        self._path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_completed_task(self, task: dict[str, Any]) -> None:
        with self._lock:
            tasks = self._load()
            task["created_at"] = datetime.now().isoformat(timespec="seconds")
            tasks.insert(0, task)
            self._save(tasks)

    def list_tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._load()

