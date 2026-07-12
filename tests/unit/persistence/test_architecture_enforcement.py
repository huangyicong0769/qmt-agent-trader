from pathlib import Path

from qmt_agent_trader.persistence.architecture import scan_forbidden_persistence


def test_architecture_scan_catches_fixture_and_passes_production(tmp_path: Path) -> None:
    fixture = tmp_path / "bad.py"
    fixture.write_text('import pandas as pd\npd.DataFrame().to_parquet("bad.parquet")\n')
    assert scan_forbidden_persistence(tmp_path)
    production = Path(__file__).parents[3] / "src/qmt_agent_trader"
    assert scan_forbidden_persistence(production) == []


def test_architecture_scan_is_alias_mode_aware_and_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "modes.py").write_text(
        "from duckdb import connect as dc\n"
        "from pathlib import Path\n"
        'open("a", mode="ab")\n'
        'Path("b").open(mode="a")\n'
        'dc("db")\n'
    )
    (tmp_path / "syntax.py").write_text("def broken(:\n")

    violations = scan_forbidden_persistence(tmp_path)
    primitives = {item.primitive for item in violations}
    assert {
        'open(..., "a")',
        'Path.open(..., "a")',
        "duckdb.connect",
        "invalid_python",
    } <= primitives


def test_architecture_scan_detects_duckdb_module_alias_connect(tmp_path: Path) -> None:
    (tmp_path / "alias.py").write_text('import duckdb as ddb\nddb.connect("db")\n')

    violations = scan_forbidden_persistence(tmp_path)

    assert any(item.primitive == "duckdb.connect" for item in violations)


def test_architecture_scan_rejects_cwd_persistence_root(tmp_path: Path) -> None:
    (tmp_path / "roots.py").write_text(
        'from pathlib import Path\nroot = Path("reports/research")\n'
    )

    primitives = {item.primitive for item in scan_forbidden_persistence(tmp_path)}

    assert "cwd_relative_persistence_root" in primitives
