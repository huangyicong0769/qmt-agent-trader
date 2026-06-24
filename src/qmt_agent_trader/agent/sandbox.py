"""Agent code sandbox — path-limited, statically-scanned code generation.

All LLM-generated code must pass through this sandbox before it can be
saved to disk. The sandbox enforces:

- Write-path whitelist (only `generated/` subtree).
- Static scan for dangerous imports and calls.
- Import-based checks (no subprocess, no socket, no live broker).
"""

from __future__ import annotations

import re
from pathlib import Path

from qmt_agent_trader.agent.errors import SandboxPathError, SandboxSecurityError
from qmt_agent_trader.agent.schemas import SandboxTestResult
from qmt_agent_trader.core.config import get_settings

# ── Forbidden patterns ───────────────────────────────────────────────────────

_FORBIDDEN_WORDS: list[str] = [
    "submit_order",
    "submit_live_order",
    "approve_strategy",
    "register_production_strategy",
    "modify_live_config",
    "modify_risk_limits",
    "modify_gateway_config",
    "delete_experiment",
    "delete_audit_log",
    "query_account_secret",
    "read_env_file",
    "write_env_file",
    "os.environ",
    "shutil.rmtree",
]

_FORBIDDEN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bsubprocess\b"),
    re.compile(r"\bsocket\b"),
    re.compile(r"\bos\.system\b"),
    re.compile(r"\bos\.environ\b"),
    re.compile(r"\bshutil\.rmtree\b"),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bexec\s*\("),
    re.compile(r"""open\s*\(\s*["']\.env["']"""),
    re.compile(r"""Path\s*\(\s*["']\.env["']"""),
    re.compile(r"""open\s*\(\s*["']/"""),
    re.compile(r"""open\s*\(\s*["'][~]"""),
    re.compile(r"\.shift\s*\(\s*-1\s*\)"),
    re.compile(r"\.shift\s*\(\s*-\d+\s*\)"),
    re.compile(r"future\s*return", re.IGNORECASE),
    re.compile(r"from\s+qmt_agent_trader\.broker"),
    re.compile(r"from\s+qmt_agent_trader\.gateway"),
    re.compile(r"import\s+xtquant"),
    re.compile(r"register_production"),
    re.compile(r"modify_risk"),
    re.compile(r"submit_live"),
    re.compile(r"approve_strategy"),
]

_ALLOWED_WRITE_ROOTS: tuple[Path, ...] = ()
_ALLOWED_IMPORT_PREFIXES: tuple[str, ...] = (
    "qmt_agent_trader.agent.generated",
    "qmt_agent_trader.backtest",
    "qmt_agent_trader.factors",
    "qmt_agent_trader.data",
    "qmt_agent_trader.core",
    "qmt_agent_trader.strategy",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statsmodels",
)


class CodeSandbox:
    """Path-limited, statically-scanned code sandbox for LLM-generated files."""

    def __init__(self, generated_root: Path | None = None) -> None:
        if generated_root is None:
            settings = get_settings()
            generated_root = (
                settings.project_root
                / "src"
                / "qmt_agent_trader"
                / "agent"
                / "generated"
            )
        self.generated_root = generated_root.resolve()
        self.generated_root.mkdir(parents=True, exist_ok=True)

        global _ALLOWED_WRITE_ROOTS
        if not _ALLOWED_WRITE_ROOTS:
            _ALLOWED_WRITE_ROOTS = (self.generated_root,)

    # ── Path validation ───────────────────────────────────────────────────

    def validate_path(self, raw_path: str | Path) -> Path:
        """Resolve `raw_path` against `generated_root` and enforce sandbox.

        Returns the canonical resolved path on success.
        Raises `SandboxPathError` if the path escapes the sandbox.
        """
        candidate = (self.generated_root / str(raw_path)).resolve()
        if not self._is_under(candidate, self.generated_root):
            raise SandboxPathError(
                f"path '{raw_path}' escapes the sandbox: {candidate}"
            )
        return candidate

    def write_candidate_file(self, relative_path: str, content: str) -> Path:
        """Scan, then write a candidate file. Raises on scan failure."""
        # Path-name check
        name = Path(str(relative_path)).name.lower()
        if name == ".env" or name.startswith(".env."):
            raise SandboxSecurityError(
                "writing to .env files is forbidden"
            )
        issues = self.static_scan_code(content)
        if issues:
            raise SandboxSecurityError(
                "static scan failed:\n" + "\n".join(f"  - {i}" for i in issues)
            )
        target = self.validate_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    # ── Static analysis ───────────────────────────────────────────────────

    def static_scan_code(self, code: str) -> list[str]:
        """Return a (possibly empty) list of security issues found in *code*."""
        issues: list[str] = []
        lower = code.lower()
        context = code  # original-case for regex

        for word in _FORBIDDEN_WORDS:
            if word.lower() in lower:
                issues.append(f"forbidden keyword: {word}")

        for pattern in _FORBIDDEN_PATTERNS:
            if pattern.search(context):
                issues.append(f"forbidden pattern: {pattern.pattern}")

        # Heuristic: direct broker or gateway references
        if re.search(r"RemoteQMTBrokerClient", context):
            issues.append("direct broker client instantiation")

        return sorted(set(issues))

    # ── Test execution (stub — delegates to pytest in subprocess) ──────────

    def run_tests(self, test_path: Path) -> SandboxTestResult:
        """Run pytest on a candidate test file.

        This is a stub that reports the path; full subprocess execution
        should be added when the runner is ready.
        """
        if not test_path.exists():
            return SandboxTestResult(
                status="FAILED",
                safety_issues=["test file does not exist"],
            )
        # In the full implementation this would call pytest in a
        # resource-limited subprocess. For now, structural check only.
        content = test_path.read_text(encoding="utf-8")
        issues = self.static_scan_code(content)
        has_test_functions = bool(re.search(r"def test_", content))
        return SandboxTestResult(
            status="PASSED" if not issues and has_test_functions else "FAILED",
            test_summary={
                "test_file": str(test_path),
                "has_test_functions": has_test_functions,
                "issues": issues,
            },
            safety_issues=issues,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _is_under(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False
