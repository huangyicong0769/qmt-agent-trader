"""Persistent chat-session repository."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from qmt_agent_trader.persistence.repositories.versioned_record import VersionedRecordRepository
from qmt_agent_trader.web.schemas import ChatSession


class ChatSessionRepository:
    def __init__(
        self, root: Path, *, locks_root: Path | None = None, quarantine_root: Path | None = None
    ) -> None:
        self.records = VersionedRecordRepository(
            root,
            ChatSession,
            store_name="chat_sessions",
            locks_root=locks_root,
            quarantine_root=quarantine_root,
            identity=lambda session: session.session_id,
        )
        self.last_diagnostics: list[Any] = []

    def create(self, session: ChatSession) -> ChatSession:
        return self.records.create(session.session_id, session)

    def get(self, session_id: str) -> ChatSession | None:
        try:
            return self.records.load(session_id)
        except FileNotFoundError:
            return None

    def list(self) -> list[ChatSession]:
        sessions, self.last_diagnostics = self.records.list_with_diagnostics()
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def update(
        self,
        session_id: str,
        operation: Callable[[ChatSession], ChatSession],
        *,
        expected_revision: int | None = None,
    ) -> ChatSession:
        return self.records.mutate(
            session_id, operation, expected_revision=expected_revision
        )

    def save(
        self, session: ChatSession, *, expected_revision: int | None = None
    ) -> ChatSession:
        return self.records.upsert(
            session.session_id,
            lambda _current: session,
            expected_revision=expected_revision,
        )

    def delete(self, session_id: str) -> bool:
        return self.records.delete(session_id)
