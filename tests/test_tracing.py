"""Tests for tracing module — no-op tracer and generation handle."""

import os
from unittest.mock import MagicMock, patch

from policy_arena.tracing import (
    SimulationTracer,
    _GenerationHandle,
    _langfuse_available,
    _NoOpTracer,
)


class TestGenerationHandle:
    def test_noop_handle(self):
        """Handle with None generation is no-op."""
        handle = _GenerationHandle(None)
        assert handle.is_active is False
        handle.end(output="test")  # should not raise

    def test_active_handle(self):
        gen = MagicMock()
        handle = _GenerationHandle(gen)
        assert handle.is_active is True
        handle.end(output="result", metadata={"key": "val"})
        gen.update.assert_called_once_with(output="result", metadata={"key": "val"})
        gen.end.assert_called_once()

    def test_end_without_metadata(self):
        gen = MagicMock()
        handle = _GenerationHandle(gen)
        handle.end(output="result")
        gen.update.assert_called_once_with(output="result")


class TestNoOpTracer:
    def test_trace_id_is_none(self):
        tracer = _NoOpTracer()
        assert tracer.trace_id is None

    def test_all_methods_are_noop(self):
        tracer = _NoOpTracer()
        tracer.start_round(1)
        tracer.end_round(results={"test": 1})
        gen = tracer.start_generation("agent_a", model="test")
        assert gen.is_active is False
        tracer.log_event("test_event", metadata={"key": "val"})
        tracer.finish(metadata={"final": True})


class TestLangfuseAvailable:
    @patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""})
    def test_not_available_when_empty(self):
        assert _langfuse_available() is False

    @patch.dict(
        os.environ,
        {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"},
    )
    def test_available_when_set(self):
        assert _langfuse_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_when_missing(self):
        # Remove keys if they exist
        os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
        os.environ.pop("LANGFUSE_SECRET_KEY", None)
        assert _langfuse_available() is False


class TestSimulationTracerCreate:
    @patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""})
    def test_creates_noop_when_no_langfuse(self):
        tracer = SimulationTracer.create(
            game_id="test",
            agent_labels=["a1"],
            agent_brains=["brain1"],
            n_rounds=10,
        )
        assert isinstance(tracer, _NoOpTracer)
        assert tracer.trace_id is None


class TestSimulationTracerMethods:
    """Test tracer methods with mocked internals."""

    def test_trace_id_with_root_span(self):
        root_span = MagicMock()
        root_span.trace_id = "test-trace-123"
        tracer = SimulationTracer(langfuse=MagicMock(), root_span=root_span)
        assert tracer.trace_id == "test-trace-123"

    def test_trace_id_none_when_no_span(self):
        tracer = SimulationTracer(langfuse=MagicMock(), root_span=None)
        assert tracer.trace_id is None

    def test_start_and_end_round(self):
        root_span = MagicMock()
        round_span = MagicMock()
        root_span.start_span.return_value = round_span
        tracer = SimulationTracer(langfuse=MagicMock(), root_span=root_span)

        tracer.start_round(1, metadata={"extra": True})
        root_span.start_span.assert_called_once()

        tracer.end_round(results={"payoffs": [1, 2]})
        round_span.create_event.assert_called_once()
        round_span.end.assert_called_once()

    def test_finish(self):
        root_span = MagicMock()
        langfuse = MagicMock()
        tracer = SimulationTracer(langfuse=langfuse, root_span=root_span)

        tracer.finish(metadata={"total_rounds": 10})
        root_span.update.assert_called_once()
        root_span.end.assert_called_once()
        langfuse.flush.assert_called_once()

    def test_inactive_after_finish(self):
        tracer = SimulationTracer(langfuse=MagicMock(), root_span=MagicMock())
        tracer.finish()
        # After finish, start_round should be a no-op
        tracer.start_round(99)

    def test_log_event(self):
        root_span = MagicMock()
        tracer = SimulationTracer(langfuse=MagicMock(), root_span=root_span)
        tracer.log_event("test_event", metadata={"key": "val"})
        root_span.create_event.assert_called_once()

    def test_start_generation(self):
        root_span = MagicMock()
        observation = MagicMock()
        root_span.start_observation.return_value = observation
        tracer = SimulationTracer(langfuse=MagicMock(), root_span=root_span)

        gen = tracer.start_generation("agent_1", model="gpt-4")
        assert gen.is_active is True
        root_span.start_observation.assert_called_once()
