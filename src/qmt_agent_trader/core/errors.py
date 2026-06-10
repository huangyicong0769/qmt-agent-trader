"""Domain exceptions."""


class QMTAgentTraderError(Exception):
    """Base error for the project."""


class ConfigurationError(QMTAgentTraderError):
    """Raised when configuration is invalid."""


class SecurityError(QMTAgentTraderError):
    """Raised when authentication or signing fails."""


class PermissionDeniedError(QMTAgentTraderError):
    """Raised when an agent or strategy attempts a forbidden action."""


class RiskCheckError(QMTAgentTraderError):
    """Raised when a risk check rejects an operation."""


class ApprovalError(QMTAgentTraderError):
    """Raised when strategy or order approval is invalid."""


class LeakageError(QMTAgentTraderError):
    """Raised when a backtest uses data that should not be visible."""
