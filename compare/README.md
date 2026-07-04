# Tool-Eval Comparison Helpers

Small helpers for turning two `tool-eval-bench` Markdown reports into a browser-friendly head-to-head HTML comparison.

The preferred interface is the main `tool-eval-bench compare-report` CLI subcommand. The scripts in this directory remain as compatibility wrappers for publishing or reviewing model-vs-model results after benchmark runs have already completed.

## Files

| File | Purpose |
|---|---|
| `compare_tool_eval.py` | Compares two single-run tool-call benchmark Markdown reports. |
| `compare_summary.py` | Compares two cross-trial summary Markdown reports. Use this for `--trials` outputs. |
| `git/` | Example published comparison artifacts: two summary reports, one generated HTML page, and a public-facing README. |

Both scripts generate self-contained HTML content, but the page loads Tailwind CSS, Font Awesome, and Google Fonts from CDNs when viewed.

## Which Script To Use

Use `compare_tool_eval.py` when each input file starts with:

```markdown
# Tool-Call Benchmark — <model>
```

Use `compare_summary.py` when each input file starts with:

```markdown
# Cross-Trial Summary — <model>
```

The summary comparer is the better default for serious comparisons because it includes trial-level reliability data such as Pass@k, Pass^k, flaky scenarios, never-pass scenarios, safety warnings, mean score, and score variance.

## Usage

Use the main CLI for either single-run reports or cross-trial summary reports. The report type is auto-detected from the Markdown heading:

```bash
tool-eval-bench compare-report \
  path/to/model_a_summary.md \
  path/to/model_b_summary.md \
  -o model_a_vs_model_b.html
```

The compatibility wrapper scripts can still be run from this directory with the project virtualenv Python:

```bash
../.venv/bin/python compare_summary.py \
  path/to/model_a_summary.md \
  path/to/model_b_summary.md \
  model_a_vs_model_b.html
```

For single-run reports:

```bash
../.venv/bin/python compare_tool_eval.py \
  path/to/model_a_report.md \
  path/to/model_b_report.md \
  model_a_vs_model_b.html
```

Then open the generated report in a browser:

```bash
xdg-open model_a_vs_model_b.html
```

## Input Expectations

The parsers are intentionally simple and expect the Markdown structure produced by `tool-eval-bench`.

`compare_tool_eval.py` reads:

- report title, run ID, date, version, backend, model, temperature, and thinking metadata
- final score, total points, rating, quality, responsiveness, deployability, and median turn time
- `## Category Scores`
- `## Scenario Results`
- `## Performance by Difficulty`
- safety-critical warning blocks, when present

`compare_summary.py` reads:

- summary title, run ID, date, version, trial count, backend, model, temperature, and thinking metadata
- mean final score, score standard deviation, mean points, rating, quality, responsiveness, deployability, and median turn time
- safety warnings across trials
- Pass@k, Pass^k, reliability gap, and 95% confidence interval
- `## Category Variance`
- `## Per-Scenario Results`
- `### Never Passes`, `### Flaky`, and `### Consistently Partial` sections, when present

If the upstream report format changes, update the regexes in the parser section of the relevant script.

## Output Behavior

Each script:

- parses both Markdown files
- selects the winner by final score or mean score
- builds a light-themed comparison page
- writes the requested HTML file
- prints the parsed model names and scores to stdout

The generated HTML includes:

- winner and runner-up cards
- key metrics table
- category comparison
- reliability and safety sections
- failure and partial-result summaries
- strengths, weaknesses, and conclusion text

`compare_summary.py` also detects likely infrastructure-failed runs where all scenarios scored zero due to server, timeout, or connection errors, and displays an explicit warning in the generated report.

## Example Artifacts

The `git/` folder contains a complete published example:

```text
git/
├── Agents-A1-Q8_0_summary.md
├── Qwen3.6-35B-A3B-UD-Q8_K_XL_summary.md
├── Agents-A1_vs_Qwen3.6-35B-A3B-UD-Q8_K_XL.html
└── README.md
```

Regenerate that style of report with:

```bash
tool-eval-bench compare-report \
  git/Agents-A1-Q8_0_summary.md \
  git/Qwen3.6-35B-A3B-UD-Q8_K_XL_summary.md \
  -o git/Agents-A1_vs_Qwen3.6-35B-A3B-UD-Q8_K_XL.html
```

Or with the compatibility wrapper:

```bash
../.venv/bin/python compare_summary.py \
  git/Agents-A1-Q8_0_summary.md \
  git/Qwen3.6-35B-A3B-UD-Q8_K_XL_summary.md \
  git/Agents-A1_vs_Qwen3.6-35B-A3B-UD-Q8_K_XL.html
```

## Limitations

- The scripts are format-specific regex parsers, not general Markdown parsers.
- Missing or renamed report sections usually produce empty table values instead of hard failures.
- The HTML depends on external CDNs for styling and icons.
- The scripts do not query SQLite or run benchmarks; they only compare existing Markdown artifacts.
- Winner selection is score-based only. Human judgment is still needed for deployment decisions, especially when safety warnings or infrastructure failures are present.
