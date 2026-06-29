"""Session-scoped todo-list state for agent runs."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from qmt_agent_trader.core.ids import new_id, shanghai_now_iso

MAX_TODO_ITEMS = 50
MAX_TODO_TITLE_LENGTH = 200


class TodoStatus(StrEnum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"

    @classmethod
    def values(cls) -> set[str]:
        return {cls.PENDING, cls.IN_PROGRESS, cls.COMPLETED, cls.BLOCKED}

    @classmethod
    def normalize(cls, value: str | TodoStatus) -> TodoStatus:
        raw = str(value)
        if raw not in cls.values():
            raise ValueError(f"unknown todo status: {raw}")
        return cls(raw)


class TodoItem(BaseModel):
    item_id: str = Field(default_factory=lambda: new_id("todo"))
    title: str
    status: TodoStatus = TodoStatus.PENDING
    notes: str = ""
    created_at: str = Field(default_factory=shanghai_now_iso)
    updated_at: str = Field(default_factory=shanghai_now_iso)


class TodoListRecord(BaseModel):
    session_id: str
    goal: str | None = None
    items: list[TodoItem] = Field(default_factory=list)
    updated_at: str = Field(default_factory=shanghai_now_iso)

    @property
    def summary(self) -> dict[str, int]:
        counts = {
            "total": len(self.items),
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "blocked": 0,
        }
        for item in self.items:
            if item.status == TodoStatus.PENDING:
                counts["pending"] += 1
            elif item.status == TodoStatus.IN_PROGRESS:
                counts["in_progress"] += 1
            elif item.status == TodoStatus.COMPLETED:
                counts["completed"] += 1
            elif item.status == TodoStatus.BLOCKED:
                counts["blocked"] += 1
        return counts

    @property
    def active_item(self) -> TodoItem | None:
        for item in self.items:
            if item.status == TodoStatus.IN_PROGRESS:
                return item
        return None

    def to_payload(self, *, include_completed: bool = True) -> dict[str, Any]:
        items = [
            item
            for item in self.items
            if include_completed or item.status != TodoStatus.COMPLETED
        ]
        active = self.active_item
        return {
            "session_id": self.session_id,
            "goal": self.goal,
            "items": [item.model_dump(mode="json") for item in items],
            "summary": self.summary,
            "active_item": active.model_dump(mode="json") if active else None,
            "updated_at": self.updated_at,
        }


class TodoListStore:
    """JSON-file store keyed by a hashed session id."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for_session(self, session_id: str) -> Path:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
        return self.root / f"{digest}.json"

    def get(self, session_id: str) -> TodoListRecord:
        path = self.path_for_session(session_id)
        if not path.exists():
            return TodoListRecord(session_id=session_id)
        return TodoListRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def replace_items(
        self,
        session_id: str,
        items: list[dict[str, Any]],
        *,
        goal: str | None = None,
    ) -> TodoListRecord:
        self._validate_item_count(len(items))
        now = shanghai_now_iso()
        record = TodoListRecord(
            session_id=session_id,
            goal=goal,
            items=[
                TodoItem(
                    title=self._clean_title(item.get("title")),
                    notes=str(item.get("notes", "")),
                    status=TodoStatus.normalize(str(item.get("status", TodoStatus.PENDING))),
                    updated_at=now,
                )
                for item in items
            ],
            updated_at=now,
        )
        return self._write(self._normalize_active(record))

    def add_item(
        self,
        session_id: str,
        *,
        title: str,
        notes: str = "",
    ) -> TodoListRecord:
        record = self.get(session_id)
        self._validate_item_count(len(record.items) + 1)
        record.items.append(TodoItem(title=self._clean_title(title), notes=notes))
        return self._write(record)

    def update_item(
        self,
        session_id: str,
        item_id: str,
        *,
        status: TodoStatus | str | None = None,
        title: str | None = None,
        notes: str | None = None,
    ) -> TodoListRecord:
        record = self.get(session_id)
        item = self._find_item(record, item_id)
        updates: dict[str, Any] = {"updated_at": shanghai_now_iso()}
        if status is not None:
            updates["status"] = TodoStatus.normalize(status)
        if title is not None:
            updates["title"] = self._clean_title(title)
        if notes is not None:
            updates["notes"] = notes
        index = record.items.index(item)
        record.items[index] = item.model_copy(update=updates)
        if updates.get("status") == TodoStatus.IN_PROGRESS:
            record.items = [
                existing.model_copy(
                    update={
                        "status": TodoStatus.PENDING,
                        "updated_at": shanghai_now_iso(),
                    }
                )
                if existing.item_id != item_id
                and existing.status == TodoStatus.IN_PROGRESS
                else existing
                for existing in record.items
            ]
        record.updated_at = shanghai_now_iso()
        return self._write(self._normalize_active(record))

    def clear_completed(self, session_id: str) -> TodoListRecord:
        record = self.get(session_id)
        record.items = [
            item for item in record.items if item.status != TodoStatus.COMPLETED
        ]
        record.updated_at = shanghai_now_iso()
        return self._write(record)

    def _write(self, record: TodoListRecord) -> TodoListRecord:
        path = self.path_for_session(record.session_id)
        path.write_text(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return record

    @staticmethod
    def _normalize_active(record: TodoListRecord) -> TodoListRecord:
        active_seen = False
        normalized: list[TodoItem] = []
        for item in record.items:
            if item.status == TodoStatus.IN_PROGRESS:
                if active_seen:
                    item = item.model_copy(
                        update={
                            "status": TodoStatus.PENDING,
                            "updated_at": shanghai_now_iso(),
                        }
                    )
                active_seen = True
            normalized.append(item)
        record.items = normalized
        return record

    @staticmethod
    def _find_item(record: TodoListRecord, item_id: str) -> TodoItem:
        for item in record.items:
            if item.item_id == item_id:
                return item
        raise ValueError(f"todo item not found: {item_id}")

    @staticmethod
    def _validate_item_count(count: int) -> None:
        if count > MAX_TODO_ITEMS:
            raise ValueError("todo list supports at most 50 items")

    @staticmethod
    def _clean_title(raw: object) -> str:
        title = str(raw or "").strip()
        if not 1 <= len(title) <= MAX_TODO_TITLE_LENGTH:
            raise ValueError("todo title must be 1-200 characters")
        return title
