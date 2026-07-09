from __future__ import annotations

from pathlib import Path


def test_removed_legacy_theme_builders_are_not_referenced_in_source() -> None:
    source = "\n".join(_source_files_text())

    removed_ontology = "THEME" + "_INDUSTRY_ONTOLOGY"
    removed_builder = "build_" + "theme_universe"
    removed_backtest = "_resolve_" + "cyclical_symbols_for_backtest"
    removed_prompt = "filters={'" + "theme':'cyclical'}"

    assert removed_ontology not in source
    assert removed_builder not in source
    assert removed_backtest not in source
    assert removed_prompt not in source


def _source_files_text() -> list[str]:
    root = Path("src/qmt_agent_trader")
    texts: list[str] = []
    for path in root.rglob("*.py"):
        if "agent/generated" in path.as_posix():
            continue
        texts.append(path.read_text(encoding="utf-8"))
    return texts
