"""GSM8K benchmark plugin — orchestrates the grade-school math evaluation.

Implements ``BenchmarkPlugin`` to:
1. Load the GSM8K test split (downloading from HuggingFace on first use)
2. Send each question to the model via the adapter (no tools)
3. Extract and compare numeric answers
4. Produce accuracy % and per-question results
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

from tool_eval_bench.adapters.base import BackendAdapter
from tool_eval_bench.domain.plugin import BenchmarkPlugin, BenchmarkResult, OnPluginProgress
from tool_eval_bench.plugins.gsm8k.dataset import GSM8KItem, load_dataset
from tool_eval_bench.plugins.gsm8k.evaluator import evaluate_answer
from tool_eval_bench.plugins.gsm8k.prompts import build_messages

logger = logging.getLogger(__name__)


def _rating_for_accuracy(accuracy: float) -> str:
    """Map GSM8K accuracy % to a star rating."""
    if accuracy >= 90:
        return "★★★★★ Excellent"
    if accuracy >= 75:
        return "★★★★ Good"
    if accuracy >= 60:
        return "★★★ Adequate"
    if accuracy >= 40:
        return "★★ Weak"
    return "★ Poor"


class GSM8KPlugin(BenchmarkPlugin):
    """Grade School Math 8K benchmark plugin.

    Configuration is passed via ``**kwargs`` to ``run()``:

    - ``n_shots`` (int): Number of few-shot CoT examples (default: 8, max 8).
    - ``limit`` (int): Max questions to evaluate (default: 200, 0 = all).
    - ``shuffle`` (bool): Shuffle question order (default: False).
    - ``force_download`` (bool): Re-download dataset even if cached.
    """

    @property
    def name(self) -> str:
        return "gsm8k"

    @property
    def description(self) -> str:
        return "Grade School Math 8K — multi-step arithmetic reasoning accuracy"

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
        """Run GSM8K evaluation."""
        n_shots: int = kwargs.get("n_shots", 8)
        limit: int = kwargs.get("limit", 200)
        shuffle: bool = kwargs.get("shuffle", False)
        force_download: bool = kwargs.get("force_download", False)
        concurrency: int = kwargs.get("concurrency", 1)
        on_download_progress = kwargs.get("on_download_progress")
        preloaded = kwargs.get("_preloaded_items")

        # Load dataset (or use preloaded items from the CLI layer)
        if preloaded is not None:
            all_items = list(preloaded)
        else:
            all_items = load_dataset(
                force_download=force_download,
                on_progress=on_download_progress,
            )
        logger.info("Loaded %d GSM8K test questions", len(all_items))

        # Shuffle if requested (with seed for reproducibility)
        if shuffle:
            rng = random.Random(seed)
            all_items = list(all_items)
            rng.shuffle(all_items)

        # Apply limit
        items = all_items[:limit] if limit > 0 else all_items
        total = len(items)
        logger.info("Evaluating %d questions (n_shots=%d)", total, n_shots)

        # Build extra params
        extra: dict[str, Any] = {}
        if seed is not None:
            extra["seed"] = seed
        if extra_params:
            extra.update(extra_params)

        t0 = time.monotonic()
        total_tokens = 0
        item_results: list[dict[str, Any]] = []
        correct_count = 0

        if concurrency <= 1:
            # Sequential path
            for idx, item in enumerate(items):
                result = await self._evaluate_one(
                    adapter,
                    item=item,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                    n_shots=n_shots,
                    extra_params=extra or None,
                )
                item_results.append(result)
                if result["correct"]:
                    correct_count += 1
                total_tokens += result.get("tokens", 0)

                if on_progress:
                    await on_progress(idx + 1, total, result)
        else:
            # Parallel path with semaphore
            sem = asyncio.Semaphore(concurrency)
            ordered: list[dict[str, Any] | None] = [None] * total
            progress_counter = 0
            progress_lock = asyncio.Lock()

            async def _run_one(idx: int, item: GSM8KItem) -> None:
                nonlocal progress_counter
                async with sem:
                    result = await self._evaluate_one(
                        adapter,
                        item=item,
                        model=model,
                        base_url=base_url,
                        api_key=api_key,
                        temperature=temperature,
                        timeout_seconds=timeout_seconds,
                        n_shots=n_shots,
                        extra_params=extra or None,
                    )
                    ordered[idx] = result
                    if on_progress:
                        async with progress_lock:
                            progress_counter += 1
                            await on_progress(progress_counter, total, result)

            tasks = [_run_one(idx, item) for idx, item in enumerate(items)]
            gather_results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, exc in enumerate(gather_results):
                if isinstance(exc, BaseException):
                    logger.error("GSM8K question %d crashed: %s", i, exc)

            for r in ordered:
                if r is not None:
                    item_results.append(r)
                    if r["correct"]:
                        correct_count += 1
                    total_tokens += r.get("tokens", 0)

        elapsed = time.monotonic() - t0
        accuracy = (correct_count / total * 100) if total > 0 else 0.0

        return BenchmarkResult(
            plugin_name=self.name,
            score=accuracy,
            score_label=f"{accuracy:.1f}% ({correct_count}/{total})",
            rating=_rating_for_accuracy(accuracy),
            details={
                "correct": correct_count,
                "total": total,
                "accuracy": round(accuracy, 2),
                "n_shots": n_shots,
                "dataset_size": len(all_items),
                "limit": limit,
                "shuffle": shuffle,
            },
            item_results=item_results,
            metadata={
                "dataset": "openai/gsm8k",
                "split": "test",
                "n_shots": n_shots,
            },
            duration_seconds=elapsed,
            total_tokens=total_tokens,
        )

    async def _evaluate_one(
        self,
        adapter: BackendAdapter,
        *,
        item: GSM8KItem,
        model: str,
        base_url: str,
        api_key: str | None,
        temperature: float,
        timeout_seconds: float,
        n_shots: int,
        extra_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Evaluate a single GSM8K question."""
        messages = build_messages(item.question, n_shots=n_shots)

        try:
            result = await adapter.chat_completion(
                model=model,
                messages=messages,
                tools=None,
                temperature=temperature,
                max_tokens=1024,
                timeout_seconds=timeout_seconds,
                api_key=api_key,
                base_url=base_url,
                extra_params=extra_params,
            )
            response_text = result.content or ""
            tokens = (result.prompt_tokens or 0) + (result.completion_tokens or 0)
        except Exception as exc:
            logger.warning("Error on question %d: %s", item.index, exc)
            return {
                "index": item.index,
                "question": item.question[:200],
                "correct": False,
                "ground_truth": item.ground_truth,
                "extracted_answer": None,
                "extraction_method": "error",
                "error": str(exc),
                "tokens": 0,
            }

        eval_result = evaluate_answer(response_text, item.ground_truth)

        return {
            "index": item.index,
            "question": item.question[:200],
            "correct": eval_result.correct,
            "ground_truth": eval_result.ground_truth,
            "extracted_answer": eval_result.extracted_answer,
            "extraction_method": eval_result.extraction_method,
            "model_response": response_text[:500],  # Truncate for storage
            "tokens": tokens,
        }

    def render_report_section(self, result: BenchmarkResult) -> list[str]:
        """Render GSM8K results as Markdown."""
        details = result.details
        md: list[str] = [
            "## GSM8K — Grade School Math",
            "",
            f"- **Accuracy**: **{result.score:.1f}%** ({details['correct']}/{details['total']})",
            f"- **Rating**: {result.rating}",
            f"- **Few-shot examples**: {details.get('n_shots', 8)}-shot CoT",
            f"- **Dataset**: openai/gsm8k test ({details.get('dataset_size', '?')} total, "
            f"{details['total']} evaluated)",
            f"- **Duration**: {result.duration_seconds:.1f}s",
        ]
        if result.total_tokens > 0:
            md.append(f"- **Tokens consumed**: {result.total_tokens:,}")
        md.append("")

        # Extraction method breakdown
        methods: dict[str, int] = {}
        for item in result.item_results:
            m = item.get("extraction_method", "unknown")
            methods[m] = methods.get(m, 0) + 1
        if methods:
            md.extend([
                "### Answer Extraction Methods",
                "",
                "| Method | Count |",
                "|---|---:|",
            ])
            for method, count in sorted(methods.items(), key=lambda x: -x[1]):
                md.append(f"| {method} | {count} |")
            md.append("")

        # Failures table (up to 20)
        failures = [r for r in result.item_results if not r["correct"]]
        if failures:
            show = failures[:20]
            md.extend([
                f"### Failed Questions ({len(failures)} total, showing {len(show)})",
                "",
                "| # | Ground Truth | Extracted | Method | Response (truncated) |",
                "|---:|---:|---:|---|---|",
            ])
            for f in show:
                gt = f["ground_truth"]
                ext = f.get("extracted_answer")
                ext_str = f"{ext}" if ext is not None else "—"
                method = f.get("extraction_method", "?")
                resp = (f.get("model_response", "") or "")[:80].replace("|", "\\|").replace("\n", " ")
                md.append(f"| {f['index']} | {gt} | {ext_str} | {method} | {resp} |")
            md.append("")

        return md
