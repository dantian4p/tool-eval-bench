# Benchmark Integrity Design

## Goal

Make repeated and resumed benchmark executions trustworthy for 2.0 comparisons
without changing CLI usage.

## Design

Execution identity and comparison identity are separate concerns. `run_id` remains
a unique timestamp-based execution identifier. A new deterministic
`config_fingerprint` is computed from canonical JSON containing the complete
comparison-relevant configuration and persisted with each run.

Leaderboard grouping uses `config_fingerprint` when present. Historical rows use
a deterministic fallback derived from their stored configuration.

Resumed tool-evaluation runs reconstruct `ScenarioResult` objects from persisted
prior passes, combine them with rerun results, and invoke the existing
`score_results()` function. SQLite and Markdown reports receive the same merged
summary. SQLite conflict updates replace configuration as well as scores and
metadata.

Plugin benchmarks serialize `RunContext` with `to_dict()` and surface persistence
errors. Error injection derives per-scenario RNG offsets from SHA-256 rather than
Python's process-randomized `hash()`.

## Validation

Regression tests cover deterministic fingerprints, comparable leaderboard
grouping, resumed scoring and reports, SQLite conflict updates, plugin metadata
serialization, and stable error-injection offsets. The release gate remains:

```bash
ruff check .
.venv/bin/python -m pytest tests/ --ignore=tests/test_llama_benchy.py
```
