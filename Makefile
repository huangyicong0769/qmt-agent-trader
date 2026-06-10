.PHONY: help test lint typecheck check

help:
	uv run qmt-agent --help

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

check: lint typecheck test
