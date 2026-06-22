# tool-eval-bench Project Assessment Report

**Date:** 2026-06-18  
**Project:** tool-eval-bench (v2.0.6)  
**Competitor analyzed:** Toolery (v0.4.1, https://github.com/karolpalys/toolery)  
**Prepared for:** Project maintainers and contributors  

---

## 1. Executive Summary

`tool-eval-bench` is a mature, feature-rich benchmark platform for evaluating LLM tool-calling quality across local inference stacks (vLLM, LiteLLM, llama.cpp). It ships 84 hand-written scenarios, pluggable accuracy benchmarks (GSM8K, MMLU, IFEval), integrated throughput measurement, speculative-decoding analytics, and robust persistence/reporting.

The codebase passes lint (`ruff`) and runs **1,778 tests in ~6.5 s** with **~65% line coverage**. Overall code quality is high, architecture is well-layered, and documentation is comprehensive.

The main competitor, **Toolery**, is newer but highly focused. It offers **143 YAML-driven scenarios**, a **Textual TUI**, **deterministic primitive-based scoring**, **empirical re-tiering**, and a powerful **adapter abstraction** (`raw` / `cloud` / `hermes`) that lets users measure what the *harness* adds versus what the *model* knows.

**Bottom line:** `tool-eval-bench` is the more complete *production* tool today, but Toolery exposes several gaps worth closing—especially scenario authoring ergonomics, test coverage of the CLI, runtime analytics, and statistical rigor in run comparisons.

---

## 2. Current State Assessment

### 2.1 Strengths

| Area | Observation |
|------|-------------|
| **Scope & features** | 69 core + 15 Hard Mode scenarios, GSM8K/MMLU/IFEval plugins, throughput (legacy + llama-benchy), speculative decoding benchmark, live Prometheus dashboard, context-pressure testing. |
| **Architecture** | Strict layered design (`domain` → `evals`/`runner` → `cli`). Adapter interface is backend-agnostic. Scenarios are self-contained. |
| **Quality automation** | `ruff check .` passes. Pre-commit hooks enforce lint on commit and pytest on push. CI-quality bar documented in `AGENTS.md`. |
| **Testing** | 1,778 tests, deterministic (httpx mocks), fast runtime, coverage gate at 55%. |
| **Reporting** | Every run persists to SQLite and produces a Markdown report with full traces. Leaderboard, history, diff, and export commands. |
| **Usability** | Zero-config discovery of local servers, rich CLI help, programmatic API, JSON/JSONL progress events. |
| **Documentation** | README, architecture guide, methodology doc, changelog, security/contributing docs are all maintained. |

### 2.2 Measured Quality Metrics

```text
Source code lines:  ~28,147
Test code lines:    ~23,320
Tests passing:      1,778 / 1,778
Line coverage:      ~65% (gate: 55%)
Lint status:        clean
Runtime (test suite): ~6.5 s
```

*Coverage by module is detailed in §4.*

---

## 3. Code Quality Analysis

### 3.1 Overall Impression

Code is readable, consistently formatted, and follows the architecture rules documented in `AGENTS.md`. Recent commits show disciplined maintenance: formatting passes, lint fixes, coverage improvements, and changelog updates.

### 3.2 Notable Quality Concerns

#### A. Monolithic CLI (`src/tool_eval_bench/cli/bench.py`)

- **4,477 lines** in a single file.
- **Coverage: 24%** — the lowest of any non-trivial module.
- Handles argument parsing, adapter setup, scenario resolution, run orchestration, plugin dispatch, throughput runs, speculative decoding, live monitoring, history/export/leaderboard, and more.

**Risk:** This file is a single point of failure for most user-facing behavior. It is hard to unit-test, review, and extend safely.

#### B. Mixed coverage across core modules

| Module | Coverage | Concern |
|--------|----------|---------|
| `cli/bench.py` | 24% | Monolithic, under-tested CLI |
| `utils/metadata.py` | 29% | Backend probing/version detection largely untested |
| `runner/throughput.py` | 64% | Streaming/legacy paths, error branches not covered |
| `runner/speculative.py` | 72% | Prometheus extraction and fallback branches |
| `evals/scenarios_planning.py` | 52% | Agentic planning scenarios |
| `evals/scenarios_agentic.py` | 57% | Large agentic scenario module |
| `plugins/hf_utils.py` | 66% | HuggingFace download retry/resume logic |
| `plugins/gsm8k/dataset.py` | 49% | Dataset loading/caching |
| `plugins/mmlu/dataset.py` | 51% | Dataset loading/caching |
| `plugins/ifeval/dataset.py` | 44% | Dataset loading/caching |

**Risk:** The least-tested code is often the most environment-dependent (CLI, metadata probing, dataset download, throughput). Failures here are likely to surface in CI or on user machines rather than in local dev.

#### C. Adapter layer

`adapters/openai_compat.py` has **97% coverage** and is well-structured. However, there is **only one concrete adapter**. Backends are distinguished by a label, not by behavior. This makes it impossible to benchmark *what the harness adds* versus *what the model does*.

#### D. Scenario authoring

Scenarios are authored in Python. While this gives full expressiveness, it:
- Requires contributors to understand the internal domain model.
- Makes peer review of scenario content harder (logic + prose mixed).
- Makes empirical re-tiering and bulk edits more difficult.

---

## 4. Test Coverage Deep-Dive

### 4.1 Strong Coverage Areas

- `domain/` models and errors: 94–100%
- `runner/async_tools.py`: 99%
- `runner/judge.py`: 97%
- `storage/db.py`: 96%
- `adapters/openai_compat.py`: 97%
- Evaluator helpers and noise: 88–100%

### 4.2 Coverage Gaps to Close

1. **CLI argument parsing and scenario resolution** (`cli/bench.py`).
   - Only one `TODO` exists in the entire source tree (in `cli/bench.py`).
   - Many flag combinations (`--context-pressure-sweep`, `--resume`, `--spec-live`, plugin-only runs) are exercised only at integration/E2E level.

2. **Backend metadata probing** (`utils/metadata.py`).
   - `/version`, `/health`, `/v1/models`, Prometheus parsing, and KV-cache capping are largely untested.
   - The recent hybrid-attention fix (v2.0.6) shows this area is active and error-prone.

3. **Throughput measurement** (`runner/throughput.py`).
   - Streaming, TTFT, concurrency, and coherence checks have uncovered branches.

4. **Dataset downloaders** (`plugins/*/dataset.py`, `hf_utils.py`).
   - Retry, resume, throttling, and HF REST fallback paths are not mocked.

5. **Scenario evaluators** in `evals/scenarios_agentic.py` and `evals/scenarios_planning.py`.
   - Many evaluator branches correspond to rare partial-credit paths.

### 4.3 Test Architecture Observations

- Tests are numerous and fast; they do not require a live inference server.
- A large fraction of tests are *regression-style* (`test_v122_changes.py`, `test_v130_features.py`, `test_review_fixes.py`). This is healthy but can mask missing unit-level coverage.
- There is no explicit contract/fuzz testing of the adapter's OpenAI wire-format normalization.

---

## 5. Competitive Analysis: Toolery

### 5.1 What Toolery Does Well

| Capability | Toolery | tool-eval-bench | Implication |
|------------|---------|-----------------|-------------|
| **Scenario count** | 143 (4 tiers) | 84 (2 tiers) | Toolery tests more surface area. |
| **Scenario format** | YAML data files | Python modules | Toolery scenarios are easier to author, review, and version. |
| **Scoring primitives** | Deterministic check registry | Inline evaluator functions | Toolery's scoring is composable and self-documenting. |
| **Failure taxonomy** | `failure_kind` + budget-independent `correctness_score` | `pass/partial/fail` only | Toolery separates "did the wrong thing" from "ran out of budget". |
| **Adapter abstraction** | `raw`, `cloud`, `hermes` | Single `OpenAICompatibleAdapter` | Toolery can measure harness contribution; tool-eval-bench cannot. |
| **Empirical re-tiering** | Re-tiers scenarios by measured pass rate | Hand-assigned difficulty tiers | Toolery keeps difficulty calibrated to current models. |
| **Run comparison** | McNemar significance test | `--compare` / `--diff` diff | Toolery adds statistical rigor. |
| **TUI** | Full Textual dashboard | CLI only | Toolery is more discoverable for interactive users. |
| **Ranking** | Time-decayed, cluster-aware, use-case personas | Leaderboard by config fingerprint | Toolery supports richer ranking analytics. |
| **Golden probe** | `golden_probe.py` to prove passability | No equivalent guard | Toolery reduces false negatives from bad scenarios. |

### 5.2 Where Toolery Is Weaker

| Capability | Toolery Gap | tool-eval-bench Advantage |
|------------|-------------|---------------------------|
| **Maturity** | v0.4.1, 277 commits, 6 GitHub stars | v2.0.6, extensive real-world usage |
| **Test volume** | 257 tests | 1,778 tests |
| **Pluggable benchmarks** | None | GSM8K, MMLU, IFEval |
| **Throughput features** | Basic llama-benchy wrapper only | Legacy + llama-benchy + speculative decode + live monitor |
| **Backend breadth** | vLLM/llama.cpp/SGLang focus | vLLM, LiteLLM, llama.cpp, SGLang, Ollama, TGI |
| **Context pressure** | Prefill tokens field only | Full context-pressure sweep with prefix-cache busting |
| **Reporting** | SQLite + per-run markdown | SQLite + markdown + leaderboard + export + diff |
| **Safety gating** | Adversarial dimension, no cap | Explicit Category K safety gate that caps rating |

### 5.3 Key "Lessons" From Toolery

1. **YAML scenarios + a check registry** dramatically lower the barrier to adding scenarios and make the scoring contract explicit.
2. **Multiple adapters** (`raw` vs `cloud` vs `hermes`) let users disentangle model capability from harness capability.
3. **Failure kinds and correctness scores** make reports more actionable.
4. **Empirical re-tiering** prevents the benchmark from becoming stale as models improve.
5. **A TUI** turns the benchmark into a daily-use dashboard rather than a one-off script.
6. **Golden probes** catch scenario authoring errors before they reach users.

---

## 6. Gap Analysis

### 6.1 High-Priority Gaps

| # | Gap | Impact | Suggested First Step |
|---|-----|--------|----------------------|
| 1 | **CLI monolith** | Hard to test, extend, and review; high regression risk | Extract subcommand modules (`cli/run.py`, `cli/perf.py`, `cli/history.py`, etc.) |
| 2 | **Low CLI test coverage** | User-facing regressions slip through | Add targeted tests for scenario resolution, flag parsing, and plugin dispatch |
| 3 | **Single adapter** | Cannot measure harness contribution; limits future integrations | Introduce an adapter registry with `raw` (current) and optional `cloud`/`agent` adapters |
| 4 | **Python scenario authoring** | Contributor friction; harder to review/re-tier | Prototype a YAML scenario loader backed by the same `ScenarioDefinition` model |
| 5 | **No failure taxonomy** | Reports less actionable; partial/fail conflate causes | Capture `failure_kind` per scenario result |

### 6.2 Medium-Priority Gaps

| # | Gap | Impact | Suggested First Step |
|---|-----|--------|----------------------|
| 6 | **No empirical re-tiering** | Difficulty tiers may drift from model reality | Add a helper that bins scenarios by measured pass rate |
| 7 | **No statistical run comparison** | Hard to tell if an improvement is real | Add McNemar or paired-proportion test to `--compare` |
| 8 | **No TUI** | Less engaging for interactive users; harder to explore history | Evaluate a small Textual/Rich TUI behind an optional extra |
| 9 | **Dataset downloader coverage** | HF throttling, resume, and fallback paths untested | Add mocked HF tests for retry/resume/REST fallback |
| 10 | **Backend metadata probing** | Active, environment-dependent code is lightly tested | Add mocks for `/version`, `/health`, `/metrics`, `/v1/models` responses |

### 6.3 Lower-Priority / Nice-to-Have Gaps

- **Use-case personas** for ranking re-weighting.
- **Cluster topology tracking** for multi-node deployments.
- **Golden probe / passability guard** for new scenarios.
- **ASCII/PNG chart generation** for reports.
- **MCP bridge** to expose mock tools to MCP-aware agents.

---

## 7. Recommendations

### 7.1 Short-Term (Next 1–2 Releases)

1. **Refactor `cli/bench.py`** into smaller subcommand modules.
   - Target: reduce `bench.py` to <1,000 lines of dispatch/routing.
   - This directly enables better test coverage and safer feature additions.

2. **Add CLI unit tests** to raise `cli/bench.py` coverage from 24% to at least 50%.
   - Focus on `_resolve_scenarios`, argument validation, plugin flag dispatch, and resume logic.

3. **Stabilize backend metadata probing tests**.
   - Mock vLLM/llama.cpp/LiteLLM responses for `/version`, `/health`, `/v1/models`, and Prometheus `/metrics`.

4. **Introduce a failure taxonomy**.
   - Capture at minimum: `wrong_tool`, `wrong_args`, `missing_step`, `forbidden_action`, `budget_exceeded`, `timeout`, `connection_error`, `model_crash`.

### 7.2 Medium-Term (Next 3–6 Months)

5. **Prototype YAML scenario definitions**.
   - Keep Python scenarios working; add a YAML loader that produces the same `ScenarioDefinition` objects.
   - Migrate a small category (e.g., Category A or K) to validate the approach.

6. **Add an adapter registry**.
   - Allow `raw` (current), `cloud` (API-key gated), and optionally a local-agent adapter.
   - Store adapter name in run metadata for ranking.

7. **Implement empirical re-tiering**.
   - Provide a helper command that bins scenarios by pass rate and suggests tier adjustments.

8. **Improve statistical comparison**.
   - Add McNemar or proportion confidence intervals to `--compare` output.

9. **Close dataset-downloader coverage gaps**.
   - Mock HuggingFace responses and file-system state for resume/retry tests.

### 7.3 Long-Term (6+ Months)

10. **Build an optional TUI** for discovering endpoints, launching runs, and browsing history/rankings.
11. **Add golden-probe passability checks** for new scenarios.
12. **Support use-case personas** for re-weighted rankings.
13. **Consider a public leaderboard website** generated from the existing SQLite/Markdown artifacts.

---

## 8. What *Not* to Change

Avoid overreacting to the competitor. `tool-eval-bench` already leads in several areas that should be preserved:

- **Pluggable academic benchmarks** (GSM8K/MMLU/IFEval) — do not drop or de-emphasize.
- **Speculative-decoding analytics** (`--spec-bench`, `--spec-live`) — a genuine differentiator.
- **Context-pressure testing** — keep the prefix-cache-busting filler and sweep support.
- **Safety gating** — maintain the Category K cap.
- **Fast, deterministic test suite** — keep tests serverless and fast.
- **Clear architecture rules** — preserve the layered dependency constraints.

---

## 9. Conclusion

`tool-eval-bench` is a strong, mature project with excellent architecture, broad feature coverage, and a fast, reliable test suite. Its main risks are **CLI complexity**, **uneven test coverage**, and **missing competitive differentiators** that Toolery has demonstrated—especially YAML-driven scenarios, multiple execution adapters, failure taxonomy, empirical re-tiering, and a TUI.

The recommended path is to **protect current strengths** while **selectively borrowing from Toolery's design**: refactor the CLI, improve coverage, introduce a failure taxonomy, and prototype YAML scenarios. These changes will make the project more maintainable, more actionable for users, and harder for competitors to surpass.

---

## Appendix A: Methodology

- Ran `ruff check .` — passed.
- Ran `.venv/bin/python -m pytest tests/ --ignore=tests/test_llama_benchy.py --cov=src/tool_eval_bench --cov-report=term-missing --cov-fail-under=55` — 1,778 passed, ~65% coverage.
- Inspected source tree and recent git history.
- Fetched Toolery README, `pyproject.toml`, CLI, scorer, runner, adapter, models, CI, and changelog from GitHub.
- Line counts from `find src/tool_eval_bench -name '*.py' | xargs wc -l` and `find tests -name '*.py' | xargs wc -l`.
