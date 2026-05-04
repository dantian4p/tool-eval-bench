"""Rich Live terminal dashboard for speculative decoding stats.

Renders a continuously-updating dashboard with:
- Acceptance rate gauge with color gradient
- Per-position acceptance waterfall bar chart with decay analysis
- Throughput sparklines (rolling 60s history)
- Draft efficiency analysis & utilization gauge
- Engine status (KV cache, requests, prefix cache)
- Spec decode config info (method, draft token count)
- Cumulative session stats with session α
"""

from __future__ import annotations

import asyncio
import signal
import time
from collections import deque

import httpx
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tool_eval_bench.runner.spec_live import (
    MetricsSnapshot,
    SpecLiveDelta,
    compute_delta,
    metrics_url_from_base,
    scrape_snapshot,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HISTORY_LEN = 60  # keep 60 samples ≈ 60 seconds at 1 Hz
_POLL_INTERVAL = 1.0  # seconds between scrapes

# Sparkline block characters (⅛ blocks, bottom-up)
_SPARK_CHARS = " ▁▂▃▄▅▆▇█"

# Color thresholds for acceptance rate
_AR_COLORS = [
    (0.0, "bright_red"),
    (0.2, "red"),
    (0.35, "dark_orange"),
    (0.5, "yellow"),
    (0.65, "green_yellow"),
    (0.8, "bright_green"),
]

# Activity indicator cycle
_ACTIVITY_FRAMES = ["◆", "◇", "◈", "◇"]


def _ar_color(rate: float) -> str:
    """Return a Rich color for an acceptance rate value."""
    color = "bright_green"
    for threshold, c in _AR_COLORS:
        if rate >= threshold:
            color = c
    return color


def _gauge_bar(value: float, width: int = 40, fill: str = "━", empty: str = "╌") -> Text:
    """Render a horizontal gauge bar with color gradient."""
    filled = int(value * width)
    filled = max(0, min(width, filled))
    color = _ar_color(value)
    bar = Text()
    bar.append(fill * filled, style=f"bold {color}")
    bar.append(empty * (width - filled), style="dim")
    bar.append(f" {value * 100:5.1f}%", style=f"bold {color}")
    return bar


def _utilization_bar(value: float, width: int = 12, label: str = "") -> Text:
    """Render a compact utilization bar (0.0–1.0) with a label."""
    clamped = max(0.0, min(1.0, value))
    filled = int(clamped * width)
    color = "bright_green" if clamped < 0.5 else "yellow" if clamped < 0.8 else "bright_red"
    bar = Text()
    if label:
        bar.append(f"{label} ", style="dim")
    bar.append("▰" * filled, style=f"bold {color}")
    bar.append("▱" * (width - filled), style="dim")
    bar.append(f" {clamped * 100:.0f}%", style=f"bold {color}")
    return bar


def _mini_gauge(value: float, width: int = 12) -> Text:
    """Render a small gauge for inline use."""
    filled = int(value * width)
    filled = max(0, min(width, filled))
    color = _ar_color(value) if value <= 1.0 else "bright_red"
    bar = Text()
    bar.append("▓" * filled, style=color)
    bar.append("░" * (width - filled), style="dim")
    return bar


def _sparkline(values: list[float], width: int = 40) -> Text:
    """Render a sparkline from a list of values."""
    if not values:
        return Text("─" * width, style="dim")

    # Take last `width` values
    data = values[-width:]
    if not data:
        return Text("─" * width, style="dim")

    mn = min(data)
    mx = max(data)
    rng = mx - mn if mx > mn else 1.0

    spark = Text()
    for v in data:
        idx = int((v - mn) / rng * (len(_SPARK_CHARS) - 1))
        idx = max(0, min(len(_SPARK_CHARS) - 1, idx))
        # Color based on relative position
        if idx >= 6:
            style = "bright_green"
        elif idx >= 4:
            style = "green"
        elif idx >= 2:
            style = "yellow"
        else:
            style = "bright_red"
        spark.append(_SPARK_CHARS[idx], style=style)

    # Pad if shorter than width
    if len(data) < width:
        padding = Text("─" * (width - len(data)), style="dim")
        return Text.assemble(padding, spark)

    return spark


def _position_bars(rates: dict[int, float], max_positions: int = 8, bar_width: int = 20) -> Table:
    """Render per-position acceptance rates as a waterfall bar chart."""
    table = Table.grid(padding=(0, 1))
    table.add_column("pos", justify="right", width=3, no_wrap=True)
    table.add_column("bar", width=bar_width + 2, no_wrap=True)
    table.add_column("pct", justify="right", width=6, no_wrap=True)
    table.add_column("note", width=6, no_wrap=True)

    positions = sorted(rates.keys())[:max_positions]
    if not positions:
        return table

    # Find the tail-off point: first position where acceptance drops below 10%
    tail_off_pos: int | None = None
    for pos in positions:
        if rates[pos] < 0.10:
            tail_off_pos = pos
            break

    for pos in positions:
        rate = rates[pos]
        color = _ar_color(rate)
        filled = int(rate * bar_width)
        filled = max(0, min(bar_width, filled))

        bar = Text()
        bar.append("█" * filled, style=f"{color}")
        bar.append("░" * (bar_width - filled), style="dim")

        # Annotate significant positions
        note = Text()
        if pos == 0:
            note.append("best", style="dim")
        elif tail_off_pos is not None and pos == tail_off_pos:
            note.append("↓tail", style="dim yellow")

        table.add_row(
            Text(f"p{pos}", style="bold"),
            bar,
            Text(f"{rate * 100:5.1f}%", style=f"bold {color}"),
            note,
        )

    return table


def _position_bars_horizontal(
    rates: dict[int, float],
    max_positions: int = 16,
    inner_w: int = 80,
) -> Table:
    """Render per-position acceptance rates as horizontal inline bars.

    Automatically wraps to multiple rows when there are too many positions
    to fit in a single line (e.g., k=12 at 80 columns gets 2 rows of 6).
    """
    positions = sorted(rates.keys())[:max_positions]
    if not positions:
        return Table.grid()

    # Determine how many positions fit per row (min cell width: 14 chars)
    min_cell_w = 14  # "p0 ████ 83%" needs ~14 chars minimum
    per_row = max(1, (inner_w - 4) // min_cell_w)
    per_row = min(per_row, len(positions))  # don't exceed actual count

    # Compute bar width from actual cells-per-row
    cell_w = max(min_cell_w, (inner_w - 4) // per_row - 2)
    bar_w = max(4, cell_w - 10)  # reserve space for "p0 " + " 83%"

    table = Table.grid(padding=(0, 1), expand=True)
    for _ in range(per_row):
        table.add_column(no_wrap=True)

    # Build cells and add rows
    for row_start in range(0, len(positions), per_row):
        row_positions = positions[row_start:row_start + per_row]
        cells = []
        for pos in row_positions:
            rate = rates[pos]
            color = _ar_color(rate)
            filled = int(rate * bar_w)
            filled = max(0, min(bar_w, filled))

            cell = Text()
            cell.append(f"p{pos} ", style="bold dim")
            cell.append("\u2588" * filled, style=f"{color}")
            cell.append("\u2591" * (bar_w - filled), style="dim")
            cell.append(f" {rate * 100:.0f}%", style=f"bold {color}")
            cells.append(cell)

        # Pad with empty cells if the last row is short
        while len(cells) < per_row:
            cells.append(Text(""))

        table.add_row(*cells)
    return table


def _format_uptime(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _spec_method_label(method: str) -> tuple[str, str]:
    """Return a human-readable label and Rich style for a spec decode method."""
    labels: dict[str, tuple[str, str]] = {
        "dflash": ("Draft Flash", "bold bright_cyan"),
        "mtp": ("Multi-Token Prediction", "bold bright_yellow"),
        "eagle": ("EAGLE", "bold bright_green"),
        "eagle3": ("EAGLE-3", "bold bright_green"),
        "mlp_speculator": ("MLP Speculator", "bold bright_cyan"),
        "ngram": ("N-Gram", "bold bright_magenta"),
        "draft_model": ("Draft Model", "bold bright_cyan"),
        "unknown": ("Speculative Decoding", "bold dim"),
    }
    label, style = labels.get(method, (method.upper(), "dim"))
    return label, style


def _efficiency_insight(delta: SpecLiveDelta) -> Text:
    """Generate a one-line efficiency insight based on current metrics."""
    text = Text()

    ar = delta.cumulative_acceptance_rate
    if ar is None:
        text.append("  ℹ ", style="dim")
        text.append("awaiting acceptance data from server", style="dim italic")
        return text

    tau = delta.cumulative_acceptance_length
    win = delta.cumulative_draft_window
    method = delta.spec_method

    if ar >= 0.6:
        text.append("  ✦ ", style="bright_green")
        text.append("Excellent", style="bold bright_green")
        text.append(" — draft model is well-aligned", style="dim")
    elif ar >= 0.4:
        text.append("  ✦ ", style="yellow")
        text.append("Good", style="bold yellow")
        text.append(" — moderate acceptance, decent speedup", style="dim")
    elif ar >= 0.2:
        text.append("  ⚡ ", style="dark_orange")
        text.append("Fair", style="bold dark_orange")
        text.append(" — high waste ratio", style="dim")
    else:
        text.append("  ⚠ ", style="bright_red")
        text.append("Poor", style="bold bright_red")
        text.append(" — draft tokens mostly rejected", style="dim")

    if win is not None and tau is not None and win > 0:
        utilization = tau / win
        if utilization < 0.3:
            optimal = max(int(tau * 1.5), 2)
            nst = delta.num_spec_tokens
            current_label = f"(current: {nst})" if nst else f"(current window ≈{win:.0f})"
            text.append(
                f"\n  💡 Consider reducing num_speculative_tokens to ~{optimal} "
                f"{current_label}",
                style="dim yellow",
            )
        # Method-specific guidance
        if method == "mtp" and utilization > 0.5:
            text.append(
                f"\n  ℹ MTP benefits from native multi-token heads — "
                f"acceptance at {ar * 100:.0f}% is typical for MTP",
                style="dim cyan",
            )
        elif method == "dflash" and utilization < 0.5 and (nst := delta.num_spec_tokens) and nst > 3:
            text.append(
                f"\n  ℹ dflash with {nst} draft tokens — "
                f"try reducing for better latency/throughput balance",
                style="dim cyan",
            )

    return text


def _per_position_decay_summary(rates: dict[int, float]) -> Text | None:
    """Generate a summary of per-position acceptance decay.

    Shows: effective positions (>20% acceptance), geometric decay rate,
    and the position where most value is exhausted.
    """
    if not rates:
        return None

    positions = sorted(rates.keys())
    if len(positions) < 2:
        return None

    text = Text()

    # Effective positions: count positions with >20% acceptance
    effective = sum(1 for p in positions if rates[p] > 0.20)
    total = len(positions)

    # Find the 50% drop point (position where rate drops below half of p0)
    p0_rate = rates.get(0, rates[positions[0]])
    half_point: int | None = None
    if p0_rate > 0:
        for pos in positions[1:]:
            if rates[pos] < p0_rate * 0.5:
                half_point = pos
                break

    text.append("  ", style="")
    eff_color = "bright_green" if effective >= total * 0.6 else "yellow" if effective >= total * 0.3 else "bright_red"
    text.append(f"{effective}/{total}", style=f"bold {eff_color}")
    text.append(" effective positions (>20%)", style="dim")

    if half_point is not None:
        text.append(f"  │  50% drop at p{half_point}", style="dim")

    # Geometric decay rate between p0 and last position
    last_pos = positions[-1]
    last_rate = rates[last_pos]
    if p0_rate > 0 and last_rate > 0 and last_pos > 0:
        import math
        decay_per_pos = math.exp(math.log(last_rate / p0_rate) / last_pos)
        text.append(f"  │  γ={decay_per_pos:.2f}/pos", style="dim")

    return text


def _build_dashboard(
    delta: SpecLiveDelta | None,
    history: deque[SpecLiveDelta],
    start_time: float,
    model_name: str,
    metrics_endpoint: str,
    poll_count: int,
    baseline_snap: MetricsSnapshot | None = None,
    term_width: int = 120,
) -> Panel:
    """Build the full dashboard layout.

    Parameters
    ----------
    term_width : int
        Terminal width in columns. Used to scale all widget widths
        so the dashboard fills but never overflows the terminal.
    """
    now = time.time()
    uptime = now - start_time

    # ── Responsive width calculations ──
    # Panel border + padding eats 4 chars (│ + padding each side)
    inner_w = max(60, term_width - 4)
    # Gauge bar scales: ~40 chars at 120 cols, wider/narrower proportionally
    gauge_w = max(20, min(60, int(inner_w * 0.35)))
    # Sparkline width: use available right-column space minus labels
    spark_w = max(16, min(50, int(inner_w * 0.25)))
    # Divider
    divider_w = max(40, inner_w - 4)
    # Bottom column split: left gets ~38% for per-position + engine
    left_col_w = max(30, int(inner_w * 0.38))

    # Activity indicator
    activity = _ACTIVITY_FRAMES[poll_count % len(_ACTIVITY_FRAMES)]
    activity_color = "bright_green" if delta is not None else "yellow"

    # ── Header ──
    header = Table.grid(padding=0, expand=True)
    header.add_column("left", no_wrap=True, ratio=1)
    header.add_column("right", no_wrap=True, justify="right")

    left_text = Text()
    left_text.append(f" {activity} ", style=f"bold {activity_color}")
    left_text.append("SPECULATIVE DECODING MONITOR", style="bold bright_magenta")

    # Show spec method badge when spec decoding is detected
    if delta is not None:
        method_label, method_style = _spec_method_label(delta.spec_method)
        if method_label:
            left_text.append("  ╌╌ ", style="dim")
            left_text.append(f"⟨ {method_label} ⟩", style=method_style)

    left_text.append("\n ", style="")
    left_text.append(" ▸ ", style="dim cyan")
    left_text.append(model_name, style="bold cyan")

    # Show draft model name if Prometheus labels reveal a model different from primary
    if delta is not None and delta.model_names:
        # The primary model is usually the longest name or matches model_name.
        # Any additional model names are likely draft models.
        other_models = {m for m in delta.model_names if m != model_name}
        if other_models:
            draft_name = sorted(other_models)[0]  # pick deterministically
            left_text.append("  ← ", style="dim")
            left_text.append(draft_name, style="dim italic cyan")

    right_text = Text()
    right_text.append(f"⏱  {_format_uptime(uptime)}", style="dim")
    right_text.append("  │  ", style="dim")
    right_text.append(f"📡 {poll_count}", style="dim")
    if delta is not None and delta.num_spec_tokens is not None:
        right_text.append("  │  ", style="dim")
        right_text.append(f"k={delta.num_spec_tokens}", style="bold cyan")

    header.add_row(left_text, right_text)

    if delta is None:
        # No data yet — show waiting state
        waiting = Text.assemble(
            ("\n\n  ", ""),
            ("⏳ ", "bold yellow"),
            ("Connecting to ", ""),
            (metrics_endpoint, "bold cyan"),
            (" …\n", ""),
            ("  Waiting for speculative decoding metrics.\n", "dim"),
            ("  Make sure the server has spec decode enabled and is serving requests.\n\n", "dim"),
        )
        return Panel(
            Group(header, waiting),
            border_style="bright_magenta",
            title="[bold bright_magenta]─── ◆ spec-live ◆ ───[/]",
            subtitle="[dim italic]Ctrl+C to exit  ·  Refreshing every 1s[/]",
        )

    # ── Use CUMULATIVE rates for gauges (always meaningful) ──
    # vLLM updates Prometheus counters every ~10s, so per-interval
    # rates are zero most of the time.  Cumulative α is always valid.
    ar = delta.cumulative_acceptance_rate if delta.cumulative_acceptance_rate is not None else 0.0

    gauge_line = Text()
    gauge_line.append("\n ")
    gauge_line.append(" ◈ ACCEPTANCE RATE  ", style="bold bright_magenta")
    gauge_line.append_text(_gauge_bar(ar, width=gauge_w))

    # Annotate with τ/window utilization and inferred num_speculative_tokens
    tau = delta.cumulative_acceptance_length
    win = delta.cumulative_draft_window
    nst = delta.num_spec_tokens
    if tau and win and win > 0:
        gauge_line.append(f"  τ={tau:.1f}/{win:.0f}", style="dim")
        if nst is not None:
            gauge_line.append(f"  [k={nst}]", style="dim cyan")

    # ── Insight line ──
    insight = _efficiency_insight(delta)

    # ── Key Metrics Grid ──
    metrics = Table.grid(padding=(0, 1), expand=True)
    metrics.add_column("l1", no_wrap=True, width=15)
    metrics.add_column("v1", no_wrap=True, width=10)
    metrics.add_column("sep1", no_wrap=True, width=1)
    metrics.add_column("l2", no_wrap=True, width=15)
    metrics.add_column("v2", no_wrap=True, width=10)
    metrics.add_column("sep2", no_wrap=True, width=1)
    metrics.add_column("l3", no_wrap=True, width=15)
    metrics.add_column("v3", no_wrap=True, width=10)

    tau_str = f"{tau:.2f}" if tau is not None else "—"
    win_str = f"{win:.1f}" if win is not None else "—"
    nst_str = str(nst) if nst is not None else "—"
    # Cumulative waste
    waste = (1.0 - ar) if ar > 0 else None
    waste_str = f"{waste * 100:.1f}%" if waste is not None else "—"
    waste_color = "bright_green" if waste and waste < 0.3 else "yellow" if waste and waste < 0.6 else "bright_red"

    metrics.add_row(
        Text("  τ Acc Length", style="dim"), Text(tau_str, style="bold cyan"),
        Text("│", style="dim"),
        Text("  Draft Window", style="dim"), Text(win_str, style="bold"),
        Text("│", style="dim"),
        Text("  Spec Tokens", style="dim"),
        Text(nst_str, style="bold cyan"),
    )
    metrics.add_row(
        Text("  Accepted t/s", style="dim"), Text(f"{delta.accepted_tps:.1f}", style="bold green"),
        Text("│", style="dim"),
        Text("  Drafted t/s", style="dim"), Text(f"{delta.drafted_tps:.1f}", style="bold"),
        Text("│", style="dim"),
        Text("  Waste Ratio", style="dim"),
        Text(waste_str, style=f"bold {waste_color}" if waste is not None else "dim"),
    )
    metrics.add_row(
        Text("  Gen t/s", style="dim"), Text(f"{delta.generation_tps:.1f}", style="bold bright_green"),
        Text("│", style="dim"),
        Text("", style="dim"), Text("", style="dim"),
        Text("", style="dim"),
        Text("", style="dim"),
        Text("", style="dim"),
    )

    # ── Divider ──
    divider = Text("  " + "╶" + "─" * divider_w + "╴", style="dim bright_magenta")

    # ── Bottom Layout: two-column table ──
    bottom = Table.grid(padding=(0, 1), expand=True)
    bottom.add_column("left", width=left_col_w, no_wrap=False)
    bottom.add_column("right", ratio=1, no_wrap=False)

    # ── Left Column: Per-Position (if available) + Engine ──

    # Engine status block — bar + pct must fit in ~20 chars at narrow widths
    cache_pct = delta.gpu_cache_pct
    cache_color = "bright_green" if cache_pct < 50 else "yellow" if cache_pct < 80 else "bright_red"
    cache_bar_w = max(6, min(10, left_col_w - 20))
    cache_filled = int(cache_pct / 100 * cache_bar_w)
    cache_bar = Text()
    cache_bar.append("▰" * cache_filled, style=f"bold {cache_color}")
    cache_bar.append("▱" * (cache_bar_w - cache_filled), style="dim")
    cache_bar.append(f" {cache_pct:.0f}%", style=f"bold {cache_color}")

    engine_table = Table.grid(padding=(0, 1))
    engine_table.add_column("label", no_wrap=True, width=13)
    engine_table.add_column("value", no_wrap=True)

    engine_table.add_row(Text("KV Cache", style="dim"), cache_bar)

    prefix_pct = delta.prefix_cache_hit_pct
    prefix_color = "bright_green" if prefix_pct > 50 else "cyan" if prefix_pct > 0 else "dim"
    engine_table.add_row(
        Text("Prefix Cache", style="dim"),
        Text(f"{prefix_pct:.1f}%", style=f"bold {prefix_color}"),
    )

    run_style = "bold yellow" if delta.running_reqs > 0 else "dim"
    wait_style = "bold red" if delta.waiting_reqs > 0 else "dim"
    engine_table.add_row(
        Text("Running", style="dim"),
        Text(f"{delta.running_reqs}", style=run_style),
    )
    engine_table.add_row(
        Text("Waiting", style="dim"),
        Text(f"{delta.waiting_reqs}", style=wait_style),
    )
    engine_table.add_row(
        Text("Prompt t/s", style="dim"),
        Text(f"{delta.prompt_tps:,.0f}", style="bold cyan"),
    )

    # Session totals — show tokens since monitor launch, not server lifetime
    session_table = Table.grid(padding=(0, 2))
    session_table.add_column("label", no_wrap=True, width=15)
    session_table.add_column("value", no_wrap=True)

    if baseline_snap is not None:
        session_accepted = delta.total_accepted - int(baseline_snap.accepted_tokens)
        session_drafted = delta.total_drafted - int(baseline_snap.draft_tokens)
    else:
        session_accepted = delta.total_accepted
        session_drafted = delta.total_drafted

    session_table.add_row(
        Text("Accepted", style="dim"),
        Text(f"{session_accepted:,}", style="bold"),
    )
    session_table.add_row(
        Text("Drafted", style="dim"),
        Text(f"{session_drafted:,}", style="bold"),
    )
    session_ar = session_accepted / session_drafted if session_drafted > 0 else 0.0
    session_table.add_row(
        Text("Session α", style="dim"),
        Text(f"{session_ar * 100:.1f}%", style=f"bold {_ar_color(session_ar)}"),
    )

    engine_panel = Panel(
        Group(engine_table, Text(""), session_table),
        title="[bold]⚙ Engine & Session[/]",
        border_style="dim cyan",
        padding=(0, 1),
    )

    left_col = engine_panel

    # ── Right Column: Sparklines + Throughput History ──
    # Use cumulative α for sparklines (always available)
    ar_hist = [d.cumulative_acceptance_rate for d in history if d.cumulative_acceptance_rate is not None]
    # For throughput, use gen_tps gauge (always updated) and filter accepted to active intervals
    gen_hist = [d.generation_tps for d in history]
    acc_hist = [d.accepted_tps for d in history if d.had_activity]
    waste_hist = [1.0 - d.cumulative_acceptance_rate for d in history if d.cumulative_acceptance_rate is not None]

    spark_table = Table.grid(padding=(0, 1))
    spark_table.add_column("label", width=13, no_wrap=True)
    spark_table.add_column("spark", no_wrap=True)
    spark_table.add_column("val", width=7, justify="right", no_wrap=True)
    spark_table.add_column("range", width=16, justify="right", no_wrap=True)


    # Accept Rate sparkline
    ar_current = f"{ar * 100:.1f}%" if ar else "—"
    ar_range = ""
    if len(ar_hist) > 1:
        ar_range = f"↕{min(ar_hist) * 100:.0f}–{max(ar_hist) * 100:.0f}%"
    spark_table.add_row(
        Text("Accept Rate", style="bold"),
        _sparkline(ar_hist, width=spark_w),
        Text(ar_current, style=f"bold {_ar_color(ar)}"),
        Text(ar_range, style="dim"),
    )

    # Gen t/s sparkline
    gen_range = ""
    if len(gen_hist) > 1:
        gen_range = f"↕{min(gen_hist):.0f}–{max(gen_hist):.0f}"
    spark_table.add_row(
        Text("Gen t/s", style="bold"),
        _sparkline(gen_hist, width=spark_w),
        Text(f"{delta.generation_tps:.1f}", style="bold bright_green"),
        Text(gen_range, style="dim"),
    )

    # Accepted t/s sparkline
    acc_range = ""
    if len(acc_hist) > 1:
        acc_range = f"↕{min(acc_hist):.0f}–{max(acc_hist):.0f}"
    spark_table.add_row(
        Text("Accepted t/s", style="bold"),
        _sparkline(acc_hist, width=spark_w),
        Text(f"{delta.accepted_tps:.1f}", style="bold green"),
        Text(acc_range, style="dim"),
    )

    # Waste ratio sparkline
    waste_current = f"{waste * 100:.0f}%" if waste is not None else "—"
    waste_range = ""
    if len(waste_hist) > 1:
        waste_range = f"↕{min(waste_hist) * 100:.0f}–{max(waste_hist) * 100:.0f}%"
    waste_style = (
        f"bold {_ar_color(1.0 - waste)}" if waste is not None else "dim"
    )
    spark_table.add_row(
        Text("Waste", style="bold"),
        _sparkline(
            [1.0 - w for w in waste_hist] if waste_hist else [],
            width=spark_w,
        ),
        Text(waste_current, style=waste_style),
        Text(waste_range, style="dim"),
    )

    sparkline_panel = Panel(
        spark_table,
        title=f"[bold]📊 History ({len(history)}/{_HISTORY_LEN}s)[/]",
        border_style="bright_cyan",
        padding=(0, 1),
    )

    # Averages panel — always visible; shows 0.0 until enough data accumulates
    from statistics import mean

    avg_table = Table.grid(padding=(0, 2))
    avg_table.add_column("label", no_wrap=True, width=14)
    avg_table.add_column("value", no_wrap=True)

    avg_ar = mean(ar_hist) if ar_hist else 0.0
    avg_gen = mean(gen_hist) if gen_hist else 0.0
    avg_acc = mean(acc_hist) if acc_hist else 0.0

    avg_table.add_row(
        Text("Avg α", style="dim"),
        Text(f"{avg_ar * 100:.1f}%", style=f"bold {_ar_color(avg_ar)}"),
    )
    avg_table.add_row(
        Text("Avg Gen t/s", style="dim"),
        Text(f"{avg_gen:.1f}", style="bold bright_green"),
    )
    avg_table.add_row(
        Text("Avg Acc t/s", style="dim"),
        Text(f"{avg_acc:.1f}", style="bold green"),
    )

    avg_panel = Panel(
        avg_table,
        title="[bold]⌀ Rolling Averages[/]",
        border_style="dim cyan",
        padding=(0, 1),
    )

    right_col = Group(sparkline_panel, avg_panel)

    bottom.add_row(left_col, right_col)

    # ── Per-Position Acceptance (full-width, below the two-column layout) ──
    per_pos_panel: Panel | None = None
    if delta.per_position_rates:
        pos_bars_hz = _position_bars_horizontal(
            delta.per_position_rates, inner_w=inner_w,
        )
        # Add compact decay summary on same line
        decay_summary = _per_position_decay_summary(delta.per_position_rates)
        if decay_summary:
            pos_content = Group(pos_bars_hz, decay_summary)
        else:
            pos_content = pos_bars_hz
        per_pos_panel = Panel(
            pos_content,
            title="[bold]▊ Per-Position Acceptance[/]",
            border_style="bright_cyan",
            padding=(0, 1),
            expand=True,
        )

    # ── Final Assembly ──
    parts: list[Text | Table | Panel] = [
        header,
        gauge_line,
        insight,
        Text(""),
        metrics,
        divider,
        Text(""),
        bottom,
    ]
    if per_pos_panel is not None:
        parts.append(Text(""))
        parts.append(per_pos_panel)

    return Panel(
        Group(*parts),
        border_style="bright_magenta",
        title="[bold bright_magenta]─── ◆ spec-live ◆ ───[/]",
        subtitle="[dim italic]Ctrl+C to exit  ·  Refreshing every 1s[/]",
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------

async def run_spec_live(
    base_url: str,
    *,
    api_key: str | None = None,
    metrics_url: str | None = None,
    model_name: str = "unknown",
    poll_interval: float = _POLL_INTERVAL,
    spec_method: str | None = None,
) -> None:
    """Run the live speculative decoding monitor (blocking).

    Polls /metrics at ``poll_interval`` and renders a Rich Live dashboard.
    Press Ctrl+C to exit gracefully.

    Uses the terminal alternate screen buffer so the dashboard occupies
    the entire terminal without disturbing previous output.
    """
    import sys

    url = metrics_url or metrics_url_from_base(base_url)

    history: deque[SpecLiveDelta] = deque(maxlen=_HISTORY_LEN)
    prev_snap: MetricsSnapshot | None = None
    baseline_snap: MetricsSnapshot | None = None  # first snapshot — for session-relative counters
    start_time = time.time()
    poll_count = 0
    last_delta: SpecLiveDelta | None = None

    # Sticky gauges — vLLM resets gauge metrics to 0 between its ~10s
    # internal update intervals.  We keep the last non-zero value so the
    # dashboard doesn't flicker between real values and zero.
    _sticky_gen_tps: float = 0.0
    _sticky_prompt_tps: float = 0.0
    _sticky_gpu_cache_pct: float = 0.0
    _sticky_prefix_cache_pct: float = 0.0

    stop_event = asyncio.Event()

    def _handle_signal() -> None:
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass  # Windows

    # ── Enter alternate screen buffer ──
    # This gives us a clean, full-terminal canvas (like htop/vim).
    # The original terminal content is restored when we leave.
    sys.stdout.write("\033[?1049h")  # smcup — enter alt screen
    sys.stdout.flush()

    try:
        console = Console()

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        ) as client:
            with Live(
                _build_dashboard(None, history, start_time, model_name, url, 0, baseline_snap,
                                 term_width=console.width),
                console=console,
                refresh_per_second=2,
                transient=False,
                screen=False,  # we manage the screen ourselves
            ) as live:
                while not stop_event.is_set():
                    snap = await scrape_snapshot(client, url, api_key)
                    poll_count += 1

                    if snap is not None and (snap.has_spec_decode or snap.has_llamacpp_metrics):
                        # Capture baseline on first successful scrape (session-relative counters)
                        if baseline_snap is None:
                            baseline_snap = snap

                        if prev_snap is not None:
                            delta = compute_delta(prev_snap, snap)

                            # ── Make everything session-relative ──
                            # All cumulative metrics should reflect only what
                            # happened while the dashboard is open.
                            if baseline_snap is not None:
                                sess_accepted = snap.accepted_tokens - baseline_snap.accepted_tokens
                                sess_drafted = snap.draft_tokens - baseline_snap.draft_tokens
                                sess_drafts = snap.num_drafts - baseline_snap.num_drafts

                                # Session acceptance rate
                                if sess_drafted > 0:
                                    delta.cumulative_acceptance_rate = sess_accepted / sess_drafted
                                else:
                                    delta.cumulative_acceptance_rate = None

                                # Session acceptance length (τ)
                                if sess_drafts > 0:
                                    delta.cumulative_acceptance_length = sess_accepted / sess_drafts
                                else:
                                    delta.cumulative_acceptance_length = None

                                # Session per-position rates from counters
                                if snap.per_position_counters and baseline_snap.per_position_counters:
                                    if sess_drafts > 0:
                                        sess_rates: dict[int, float] = {}
                                        for pos, count in snap.per_position_counters.items():
                                            base_count = baseline_snap.per_position_counters.get(pos, 0.0)
                                            sess_rates[pos] = (count - base_count) / sess_drafts
                                        delta.per_position_rates = sess_rates
                                    else:
                                        # No new drafts yet — don't show stale all-time rates
                                        delta.per_position_rates = {}
                                elif sess_drafts == 0:
                                    delta.per_position_rates = {}

                            # Update sticky gauges — keep last non-zero value
                            if delta.generation_tps > 0:
                                _sticky_gen_tps = delta.generation_tps
                            if delta.prompt_tps > 0:
                                _sticky_prompt_tps = delta.prompt_tps
                            if delta.gpu_cache_pct > 0:
                                _sticky_gpu_cache_pct = delta.gpu_cache_pct
                            if delta.prefix_cache_hit_pct > 0:
                                _sticky_prefix_cache_pct = delta.prefix_cache_hit_pct

                            # Apply sticky values when current reading is zero
                            if delta.generation_tps == 0:
                                delta.generation_tps = _sticky_gen_tps
                            if delta.prompt_tps == 0:
                                delta.prompt_tps = _sticky_prompt_tps
                            if delta.gpu_cache_pct == 0:
                                delta.gpu_cache_pct = _sticky_gpu_cache_pct
                            if delta.prefix_cache_hit_pct == 0:
                                delta.prefix_cache_hit_pct = _sticky_prefix_cache_pct

                            history.append(delta)
                            last_delta = delta

                            # Override spec method if user specified --spec-method
                            if spec_method is not None:
                                delta.spec_method = spec_method
                        prev_snap = snap
                    elif snap is not None and prev_snap is None:
                        # First scrape, no spec decode counters yet — store for next
                        prev_snap = snap

                    live.update(
                        _build_dashboard(
                            last_delta, history, start_time,
                            model_name, url, poll_count,
                            baseline_snap,
                            term_width=console.width,
                        )
                    )

                    try:
                        await asyncio.wait_for(
                            stop_event.wait(), timeout=poll_interval,
                        )
                        break  # stop_event was set
                    except asyncio.TimeoutError:
                        pass  # normal: poll again

    finally:
        # ── Leave alternate screen buffer ──
        sys.stdout.write("\033[?1049l")  # rmcup — leave alt screen
        sys.stdout.flush()

    # Print session summary to the restored normal terminal
    console = Console()
    console.print()
    console.print("  [bold bright_magenta]◆ spec-live[/] stopped.")

    # Print session summary
    if history:
        ar_vals = [d.cumulative_acceptance_rate for d in history if d.cumulative_acceptance_rate is not None]
        gen_vals = [d.generation_tps for d in history]
        if ar_vals:
            from statistics import mean, stdev

            avg_ar = mean(ar_vals)
            std_ar = stdev(ar_vals) if len(ar_vals) > 1 else 0.0
            avg_gen = mean(gen_vals) if gen_vals else 0.0
            max_gen = max(gen_vals) if gen_vals else 0.0

            console.print()

            # Session-relative totals for exit summary
            if last_delta and baseline_snap:
                sess_accepted = last_delta.total_accepted - int(baseline_snap.accepted_tokens)
                sess_drafted = last_delta.total_drafted - int(baseline_snap.draft_tokens)
            elif last_delta:
                sess_accepted = last_delta.total_accepted
                sess_drafted = last_delta.total_drafted
            else:
                sess_accepted = 0
                sess_drafted = 0

            console.print(
                Panel(
                    Text.assemble(
                        ("  Duration:        ", "dim"),
                        (_format_uptime(time.time() - start_time), "bold"),
                        ("  │  ", "dim"),
                        (f"{poll_count} polls", "dim"),
                        ("\n", ""),
                        ("  Avg α:           ", "dim"),
                        (f"{avg_ar * 100:.1f}%", f"bold {_ar_color(avg_ar)}"),
                        (" ± ", "dim"),
                        (f"{std_ar * 100:.1f}%", "dim"),
                        ("\n", ""),
                        ("  Avg Gen t/s:     ", "dim"),
                        (f"{avg_gen:.1f}", "bold bright_green"),
                        ("  ", ""),
                        ("peak ", "dim"),
                        (f"{max_gen:.1f}", "bold"),
                        ("\n", ""),
                        ("  Session tokens:  ", "dim"),
                        (f"{sess_accepted:,}", "bold"),
                        (" accepted  / ", "dim"),
                        (f"{sess_drafted:,}", "bold"),
                        (" drafted", "dim"),
                    ),
                    title="[bold]Session Summary[/]",
                    border_style="bright_magenta",
                    padding=(0, 1),
                )
            )
    console.print()

