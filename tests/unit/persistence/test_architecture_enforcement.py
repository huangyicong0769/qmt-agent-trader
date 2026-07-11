from pathlib import Path

from qmt_agent_trader.persistence.architecture import scan_forbidden_persistence


def test_architecture_scan_catches_fixture_and_passes_production(tmp_path: Path) -> None:
    fixture = tmp_path / "bad.py"
    fixture.write_text('from pathlib import Path\nPath("x").write_text("bad")\n')
    assert scan_forbidden_persistence(tmp_path)
    production = Path(__file__).parents[3] / "src/qmt_agent_trader"
    assert scan_forbidden_persistence(production) == []
