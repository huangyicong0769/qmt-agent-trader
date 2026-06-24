"""Configuration helpers for the QMT Agent Studio web interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    reload: bool = False
    title: str = "QMT Agent Studio"
    data_dir: Path = Path("data")

    @property
    def artifact_roots(self) -> list[Path]:
        return [
            Path("src/qmt_agent_trader/agent/generated"),
            Path("data/reports"),
            Path("data/backtests"),
            Path("data/audit"),
        ]

    def is_path_safe(self, requested: Path) -> bool:
        """Check if a path is within allowed artifact roots."""
        resolved = requested.resolve()
        for root in self.artifact_roots:
            root_resolved = root.resolve()
            try:
                resolved.relative_to(root_resolved)
                return True
            except ValueError:
                continue
        return False


web_config = WebConfig()


def get_web_config() -> WebConfig:
    return web_config
