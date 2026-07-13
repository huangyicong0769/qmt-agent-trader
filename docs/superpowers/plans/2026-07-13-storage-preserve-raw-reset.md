# Preserve-Raw Storage Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add and execute a rollback-safe destructive reset that retains healthy raw provider data.

**Architecture:** `StorageOperations` owns deterministic planning and confirmed execution. Existing
path, lock, atomic-file, migration, and verification seams remain authoritative.

**Tech Stack:** Python, Typer, PyArrow, DuckDB, pytest, uv.

## Global Constraints

- Only `preserve-raw` is supported.
- A current dry-run digest is mandatory for execution.
- Successful execution permanently deletes staged legacy state.
- Raw file identity must remain unchanged.

## Tasks

- [x] Add deterministic, read-only reset planning and raw/symlink validation.
- [x] Add maintenance-barrier execution, staging, initialization, deep verification, receipt, and rollback.
- [x] Expose dry-run and confirmed execution through `qmt-agent storage reset`.
- [x] Cover success, drift rejection, corrupt raw, symlink safety, and injected failure rollback.
- [x] Run full repository verification, execute the real reset, and verify the clean state.
- [x] Merge the verified atomic commits to `main` and rerun the final gate.
