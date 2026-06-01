"""IFEval benchmark plugin — orchestrator and report rendering.

Implements ``BenchmarkPlugin`` for the Instruction Following Evaluation
benchmark (541 prompts, 25 constraint types).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

from tool_eval_bench.adapters.base import BackendAdapter
from tool_eval_bench.domain.plugin import (
    BenchmarkPlugin,
    BenchmarkResult,
    OnPluginProgress,
)
from tool_eval_bench.plugins.ifeval.dataset import IFEvalItem, load_dataset
from tool_eval_bench.plugins.ifeval.evaluator import evaluate_prompt

logger = logging.getLogger(__name__)


def _rating_for_accuracy(accuracy: float) -> str:
    if accuracy >= 85:
        return "★★★★★ Excellent"
    if accuracy >= 70:
        return "★★★★ Good"
    if accuracy >= 55:
        return "★★★ Adequate"
    if accuracy >= 40:
        return "★★ Weak"
    return "★ Poor"


class IFEvalPlugin(BenchmarkPlugin):
    """IFEval benchmark — 541 prompts with 25 constraint types."""

    @property
    def name(self) -> str:
        return "ifeval"

    @property
    def description(self) -> str:
        return "Instruction Following Evaluation (541 prompts, 25 constraint types)"

    async def run(
        self,
        adapter: BackendAdapter,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        seed: int | None = None,
        extra_params: dict[str, Any] | None = None,
        on_progress: OnPluginProgress | None = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Run IFEval evaluation."""
        limit: int = kwargs.get("limit", 0)
        concurrency: int = kwargs.get("concurrency", 1)
        preloaded = kwargs.get("_preloaded_items")

        # Load dataset
        if preloaded is not None:
            all_items = list(preloaded)
        else:
            on_download = kwargs.get("on_download_progress")
            all_items = load_dataset(on_progress=on_download)

        logger.info("Loaded %d IFEval prompts", len(all_items))

        if limit > 0:
            all_items = all_items[:limit]

        total = len(all_items)
        if total == 0:
            return BenchmarkResult(
                plugin_name="ifeval",
                score=0.0,
                score_label="0/0",
                rating=_rating_for_accuracy(0),
                details={"prompts_passed": 0, "total": 0},
            )

        # Evaluate
        sem = asyncio.Semaphore(concurrency)
        results: list[dict[str, Any]] = [{}] * total
        prompts_passed = 0
        instructions_passed = 0
        instructions_total = 0
        total_tokens = 0
        t_start = time.monotonic()

        async def eval_one(idx: int, item: IFEvalItem) -> None:
            nonlocal prompts_passed, instructions_passed, instructions_total
            nonlocal total_tokens

            messages = [
                {"role": "user", "content": item.prompt},
            ]

            try:
                async with sem:
                    response = await adapter.chat_completion(
                        model=model,
                        messages=messages,
                        tools=None,
                        temperature=temperature,
                        max_tokens=4096,
                        timeout_seconds=timeout_seconds,
                        api_key=api_key,
                        base_url=base_url,
                        extra_params=extra_params,
                    )

                content = response.content or ""
                total_tokens += (response.prompt_tokens or 0) + (response.completion_tokens or 0)
            except Exception as exc:
                logger.warning("Error on prompt %d: %s", item.key, exc)
                content = ""

            result = evaluate_prompt(content, item.instruction_id_list, item.kwargs)

            if result.prompt_pass:
                prompts_passed += 1
            instructions_passed += result.instructions_passed
            instructions_total += result.instructions_total

            results[idx] = {
                "key": item.key,
                "prompt": item.prompt[:200],
                "prompt_pass": result.prompt_pass,
                "instructions_passed": result.instructions_passed,
                "instructions_total": result.instructions_total,
                "instruction_details": [
                    {
                        "id": ir.instruction_id,
                        "passed": ir.passed,
                        "error": ir.error,
                    }
                    for ir in result.instruction_results
                ],
                "model_response": content[:500],
            }

            if on_progress:
                completed = sum(1 for r in results if r)
                await on_progress(completed, total, results[idx])

        tasks = [eval_one(i, item) for i, item in enumerate(all_items)]
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, exc in enumerate(gather_results):
            if isinstance(exc, BaseException):
                logger.error("IFEval prompt %d crashed: %s", i, exc)

        duration = time.monotonic() - t_start
        prompt_accuracy = prompts_passed / total * 100
        instruction_accuracy = (
            instructions_passed / instructions_total * 100
            if instructions_total > 0 else 0
        )

        # Per-constraint-type breakdown
        constraint_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"passed": 0, "total": 0}
        )
        for r in results:
            for detail in r.get("instruction_details", []):
                cid = detail["id"]
                constraint_stats[cid]["total"] += 1
                if detail["passed"]:
                    constraint_stats[cid]["passed"] += 1

        return BenchmarkResult(
            plugin_name="ifeval",
            score=round(prompt_accuracy, 2),
            score_label=(
                f"Prompt: {prompt_accuracy:.1f}% ({prompts_passed}/{total}) | "
                f"Instruction: {instruction_accuracy:.1f}% "
                f"({instructions_passed}/{instructions_total})"
            ),
            rating=_rating_for_accuracy(prompt_accuracy),
            details={
                "prompts_passed": prompts_passed,
                "total": total,
                "prompt_accuracy": round(prompt_accuracy, 2),
                "instruction_accuracy": round(instruction_accuracy, 2),
                "instructions_passed": instructions_passed,
                "instructions_total": instructions_total,
                "dataset_size": 541,
                "constraint_types": {
                    cid: {
                        "passed": s["passed"],
                        "total": s["total"],
                        "accuracy": round(s["passed"] / s["total"] * 100, 1)
                        if s["total"] else 0,
                    }
                    for cid, s in sorted(constraint_stats.items())
                },
            },
            item_results=results,
            metadata={"dataset": "google/IFEval"},
            duration_seconds=round(duration, 2),
            total_tokens=total_tokens,
        )

    def render_report_section(self, result: BenchmarkResult) -> list[str]:
        """Render Markdown report section for IFEval results."""
        d = result.details
        lines = [
            "## IFEval — Instruction Following Evaluation",
            "",
            f"**Prompt-level Accuracy:** {d.get('prompt_accuracy', 0):.1f}% "
            f"({d['prompts_passed']}/{d['total']})",
            f"**Instruction-level Accuracy:** {d.get('instruction_accuracy', 0):.1f}% "
            f"({d.get('instructions_passed', 0)}/{d.get('instructions_total', 0)})",
            f"**Rating:** {result.rating}",
            f"**Duration:** {result.duration_seconds:.1f}s",
            f"**Tokens:** {result.total_tokens:,}",
            "",
        ]

        # Per-constraint-type breakdown
        ct = d.get("constraint_types", {})
        if ct:
            lines.extend([
                "### Per-Constraint Accuracy",
                "",
                "| Constraint | Passed | Total | Accuracy |",
                "|---|---|---|---|",
            ])
            # Sort by accuracy ascending (worst first)
            sorted_ct = sorted(
                ct.items(),
                key=lambda x: x[1].get("accuracy", 0),
            )
            for cid, stats in sorted_ct:
                lines.append(
                    f"| `{cid}` | {stats['passed']} | {stats['total']} | "
                    f"{stats['accuracy']:.1f}% |"
                )
            lines.append("")

        # Failed prompts (sample)
        failures = [r for r in result.item_results if not r.get("prompt_pass")]
        if failures:
            show = failures[:15]
            header = f"{len(failures)} total"
            if len(failures) > 15:
                header += ", showing 15"
            lines.extend([
                f"### Failed Prompts ({header})",
                "",
            ])
            for f in show:
                failed_ids = [
                    d["id"] for d in f.get("instruction_details", [])
                    if not d["passed"]
                ]
                lines.append(
                    f"- **Key {f['key']}**: failed `{'`, `'.join(failed_ids)}`"
                )
            lines.append("")

        return lines
