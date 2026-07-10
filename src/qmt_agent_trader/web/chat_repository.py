"""Persistent chat-session repository."""

from pathlib import Path

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
        )

    def create(self, session: ChatSession) -> ChatSession:
        return self.records.create(session.session_id, session)

    def get(self, session_id: str) -> ChatSession | None:
        try:
            return self.records.load(session_id)
        except FileNotFoundError:
            return None

    def list(self) -> list[ChatSession]:
        sessions, _diagnostics = self.records.list_with_diagnostics()
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def update(self, session_id: str, operation):  # type: ignore[no-untyped-def]
        return self.records.mutate(session_id, operation)
