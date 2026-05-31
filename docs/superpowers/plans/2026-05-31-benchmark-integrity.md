# Benchmark Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair benchmark-integrity defects in comparison grouping, resumed runs, plugin persistence, and deterministic error injection.

**Architecture:** Keep unique execution IDs and add deterministic configuration fingerprints. Reuse `score_results()` for resumed summaries so stored and reported aggregates follow one scoring path.

**Tech Stack:** Python, SQLite, pytest, Ruff

---

### Task 1: Add regression coverage

**Files:**
- Modify: `tests/test_ids.py`
- Modify: `tests/test_leaderboard_display.py`
- Modify: `tests/test_storage_metadata.py`
- Modify: `tests/test_orchestrator.py`
- Create: `tests/test_benchmark_integrity.py`

- [ ] Add failing tests for deterministic fingerprints, strict leaderboard grouping,
  SQLite upsert config replacement, stable RNG offsets, plugin metadata conversion,
  and resumed aggregate/report reconstruction.
- [ ] Run the targeted tests and confirm failures identify the current defects.

### Task 2: Implement identity and persistence fixes

**Files:**
- Modify: `src/tool_eval_bench/utils/ids.py`
- Modify: `src/tool_eval_bench/storage/db.py`
- Modify: `src/tool_eval_bench/cli/bench.py`

- [ ] Add `build_config_fingerprint()` using canonical JSON and SHA-256.
- [ ] Replace plugin `RunContext` persistence payloads with JSON-safe dictionaries.
- [ ] Update SQLite conflicts to replace the complete persisted run row.

### Task 3: Implement scoring and comparison fixes

**Files:**
- Modify: `src/tool_eval_bench/runner/orchestrator.py`
- Modify: `src/tool_eval_bench/runner/service.py`
- Modify: `src/tool_eval_bench/cli/leaderboard.py`

- [ ] Add stable per-scenario SHA-256 RNG offsets.
- [ ] Reconstruct prior results and rescore merged resumed runs with `score_results()`.
- [ ] Persist fingerprints and group leaderboard rows by comparable configuration.

### Task 4: Update release documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `CHANGELOG.md`

- [ ] Correct test counts and document the integrity fixes.

### Task 5: Verify

- [ ] Run targeted regression tests.
- [ ] Run `ruff check .`.
- [ ] Run `.venv/bin/python -m pytest tests/ --ignore=tests/test_llama_benchy.py`.
