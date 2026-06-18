"""Tests for the failure_kind taxonomy introduced in the Unreleased refactor."""

from __future__ import annotations

import asyncio

import httpx

from tool_eval_bench.domain.scenarios import (
    FailureKind,
    ScenarioEvaluation,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)

# ---------------------------------------------------------------------------
# Domain model round-trip
# ---------------------------------------------------------------------------


class TestScenarioResultFailureKind:
    def test_to_dict_includes_failure_kind(self) -> None:
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.FAIL,
            points=0,
            summary="wrong tool",
            failure_kind=FailureKind.WRONG_TOOL,
        )
        d = result.to_dict()
        assert d["failure_kind"] == FailureKind.WRONG_TOOL

    def test_to_dict_omits_failure_kind_when_none(self) -> None:
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="ok",
            failure_kind=None,
        )
        d = result.to_dict()
        assert "failure_kind" not in d

    def test_from_dict_restores_failure_kind(self) -> None:
        data = {
            "scenario_id": "TC-01",
            "status": "fail",
            "points": 0,
            "summary": "timeout",
            "failure_kind": FailureKind.TIMEOUT,
        }
        result = ScenarioResult.from_dict(data)
        assert result.failure_kind == FailureKind.TIMEOUT

    def test_from_dict_defaults_failure_kind_to_none(self) -> None:
        data = {
            "scenario_id": "TC-01",
            "status": "pass",
            "points": 2,
            "summary": "ok",
        }
        result = ScenarioResult.from_dict(data)
        assert result.failure_kind is None


# ---------------------------------------------------------------------------
# Runtime error classification
# ---------------------------------------------------------------------------


class TestClassifyRuntimeError:
    def test_timeout(self) -> None:
        from tool_eval_bench.runner.orchestrator import _classify_runtime_error

        assert _classify_runtime_error(httpx.TimeoutException("timed out")) == FailureKind.TIMEOUT
        assert _classify_runtime_error(asyncio.TimeoutError()) == FailureKind.TIMEOUT

    def test_connection_error(self) -> None:
        from tool_eval_bench.runner.orchestrator import _classify_runtime_error

        assert (
            _classify_runtime_error(httpx.ConnectError("refused")) == FailureKind.CONNECTION_ERROR
        )

    def test_server_error(self) -> None:
        from tool_eval_bench.runner.orchestrator import _classify_runtime_error

        response = httpx.Response(500)
        exc = httpx.HTTPStatusError("server error", request=None, response=response)
        assert _classify_runtime_error(exc) == FailureKind.SERVER_ERROR

    def test_client_http_error_is_model_crash(self) -> None:
        from tool_eval_bench.runner.orchestrator import _classify_runtime_error

        response = httpx.Response(400)
        exc = httpx.HTTPStatusError("bad request", request=None, response=response)
        assert _classify_runtime_error(exc) == FailureKind.MODEL_CRASH

    def test_generic_exception_is_model_crash(self) -> None:
        from tool_eval_bench.runner.orchestrator import _classify_runtime_error

        assert _classify_runtime_error(ValueError("boom")) == FailureKind.MODEL_CRASH


# ---------------------------------------------------------------------------
# Evaluation failure classification
# ---------------------------------------------------------------------------


class TestClassifyEvaluationFailure:
    def test_pass_has_no_failure_kind(self) -> None:
        from tool_eval_bench.runner.orchestrator import _classify_evaluation_failure

        state = ScenarioState()
        evaluation = ScenarioEvaluation(status=ScenarioStatus.PASS, points=2, summary="ok")
        assert _classify_evaluation_failure(state, evaluation) is None

    def test_evaluator_failure_kind_takes_precedence(self) -> None:
        from tool_eval_bench.runner.orchestrator import _classify_evaluation_failure

        state = ScenarioState()
        evaluation = ScenarioEvaluation(
            status=ScenarioStatus.FAIL,
            points=0,
            summary="bad",
            failure_kind=FailureKind.FORBIDDEN_ACTION,
        )
        assert _classify_evaluation_failure(state, evaluation) == FailureKind.FORBIDDEN_ACTION

    def test_no_tool_calls_is_missing_step(self) -> None:
        from tool_eval_bench.runner.orchestrator import _classify_evaluation_failure

        state = ScenarioState()
        evaluation = ScenarioEvaluation(status=ScenarioStatus.FAIL, points=0, summary="did nothing")
        assert _classify_evaluation_failure(state, evaluation) == FailureKind.MISSING_STEP

    def test_forbidden_summary(self) -> None:
        from tool_eval_bench.runner.orchestrator import _classify_evaluation_failure

        state = ScenarioState()
        state.tool_calls.append(
            ToolCallRecord(id="c1", name="delete", raw_arguments="{}", arguments={}, turn=1)
        )
        evaluation = ScenarioEvaluation(
            status=ScenarioStatus.FAIL,
            points=0,
            summary="Called forbidden tool delete",
        )
        assert _classify_evaluation_failure(state, evaluation) == FailureKind.FORBIDDEN_ACTION

    def test_wrong_tool_summary(self) -> None:
        from tool_eval_bench.runner.orchestrator import _classify_evaluation_failure

        state = ScenarioState()
        state.tool_calls.append(
            ToolCallRecord(id="c1", name="get_weather", raw_arguments="{}", arguments={}, turn=1)
        )
        evaluation = ScenarioEvaluation(
            status=ScenarioStatus.FAIL,
            points=0,
            summary="used wrong tool get_weather",
        )
        assert _classify_evaluation_failure(state, evaluation) == FailureKind.WRONG_TOOL

    def test_argument_summary(self) -> None:
        from tool_eval_bench.runner.orchestrator import _classify_evaluation_failure

        state = ScenarioState()
        state.tool_calls.append(
            ToolCallRecord(id="c1", name="get_weather", raw_arguments="{}", arguments={}, turn=1)
        )
        evaluation = ScenarioEvaluation(
            status=ScenarioStatus.FAIL,
            points=0,
            summary="parameter location was wrong",
        )
        assert _classify_evaluation_failure(state, evaluation) == FailureKind.WRONG_ARGS

    def test_default_is_wrong_args_when_tools_were_called(self) -> None:
        from tool_eval_bench.runner.orchestrator import _classify_evaluation_failure

        state = ScenarioState()
        state.tool_calls.append(
            ToolCallRecord(id="c1", name="get_weather", raw_arguments="{}", arguments={}, turn=1)
        )
        evaluation = ScenarioEvaluation(
            status=ScenarioStatus.FAIL, points=0, summary="unexpected result"
        )
        assert _classify_evaluation_failure(state, evaluation) == FailureKind.WRONG_ARGS
