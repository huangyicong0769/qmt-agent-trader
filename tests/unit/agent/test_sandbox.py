"""Tests for agent.sandbox — path and security checks."""

from __future__ import annotations

import pytest

from qmt_agent_trader.agent.errors import SandboxPathError, SandboxSecurityError
from qmt_agent_trader.agent.sandbox import CodeSandbox


@pytest.fixture
def sandbox(tmp_path):
    return CodeSandbox(generated_root=tmp_path / "generated")


def test_write_inside_sandbox(sandbox: CodeSandbox) -> None:
    path = sandbox.write_candidate_file("factors/my_factor.py", "# safe code")
    assert path.exists()
    assert "generated" in str(path)


def test_write_dotenv_rejected(sandbox: CodeSandbox) -> None:
    code = "print('hello')"
    with pytest.raises(SandboxSecurityError):
        sandbox.write_candidate_file(".env", code)


def test_write_broker_rejected(sandbox: CodeSandbox) -> None:
    code = "from qmt_agent_trader.broker import RemoteQMTBrokerClient"
    with pytest.raises(SandboxSecurityError):
        sandbox.write_candidate_file("x.py", code)


def test_static_scan_subprocess(sandbox: CodeSandbox) -> None:
    issues = sandbox.static_scan_code("import subprocess\nsubprocess.run('ls')")
    assert len(issues) >= 1
    assert any("subprocess" in i for i in issues)


def test_static_scan_submit_order(sandbox: CodeSandbox) -> None:
    issues = sandbox.static_scan_code("def submit_order(): pass")
    assert any("submit_order" in i for i in issues)


def test_static_scan_os_environ(sandbox: CodeSandbox) -> None:
    issues = sandbox.static_scan_code("import os\nkey = os.environ['SECRET']")
    assert len(issues) >= 1


def test_static_scan_shift_neg_one(sandbox: CodeSandbox) -> None:
    issues = sandbox.static_scan_code("df['close'].shift(-1)")
    assert any("shift" in i for i in issues)


def test_static_scan_shift_neg_five(sandbox: CodeSandbox) -> None:
    issues = sandbox.static_scan_code("df['close'].shift(-5)")
    assert any("shift" in i for i in issues)


def test_static_scan_eval(sandbox: CodeSandbox) -> None:
    issues = sandbox.static_scan_code("eval(code)")
    assert any("eval" in i for i in issues)


def test_static_scan_exec(sandbox: CodeSandbox) -> None:
    issues = sandbox.static_scan_code("exec('import os')")
    assert any("exec" in i for i in issues)


def test_static_scan_clean_code(sandbox: CodeSandbox) -> None:
    issues = sandbox.static_scan_code(
        "import pandas as pd\n\ndef momentum_20d(df):\n    return df['close'].pct_change(20)\n"
    )
    assert issues == []


def test_validate_path_inside(sandbox: CodeSandbox) -> None:
    path = sandbox.validate_path("factors/test.py")
    assert "generated" in str(path)


def test_validate_path_escape_raises(sandbox: CodeSandbox) -> None:
    with pytest.raises(SandboxPathError):
        sandbox.validate_path("../../../.env")


def test_static_scan_broker_import(sandbox: CodeSandbox) -> None:
    issues = sandbox.static_scan_code(
        "from qmt_agent_trader.broker import RemoteQMTBrokerClient\n"
    )
    assert any("broker" in i.lower() for i in issues)


def test_safe_factor_code_passes(sandbox: CodeSandbox) -> None:
    code = '''"""A momentum factor."""
import pandas as pd

def compute(bars: pd.DataFrame) -> pd.Series:
    return bars.groupby("symbol")["close"].pct_change(20)
'''
    issues = sandbox.static_scan_code(code)
    assert issues == []
