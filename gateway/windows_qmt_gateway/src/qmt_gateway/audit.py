"""Gateway audit log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GatewayAuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"event_type": event_type, "payload": payload},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")

    def read(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        ]
