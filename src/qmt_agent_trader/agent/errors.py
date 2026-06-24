"""Agent-specific domain errors."""

from __future__ import annotations


class AgentError(Exception):
    """Base error for the Agent subsystem."""


class ToolNotFoundError(AgentError):
    """The requested tool is not registered."""


class ToolDuplicateError(AgentError):
    """A tool with this name already exists in the registry."""


class ToolExecutionError(AgentError):
    """A tool raised an error during execution."""

    def __init__(self, tool_name: str, original: Exception) -> None:
        super().__init__(f"tool '{tool_name}' failed: {original}")
        self.tool_name = tool_name
        self.original = original


class SandboxError(AgentError):
    """A sandbox constraint was violated."""


class SandboxPathError(SandboxError):
    """A write target lies outside the permitted sandbox area."""


class SandboxSecurityError(SandboxError):
    """Static scan detected forbidden patterns in generated code."""


class ExperimentNotFoundError(AgentError):
    """The requested experiment does not exist."""


class AuditError(AgentError):
    """An audit-log operation failed."""
