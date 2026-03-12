"""Langfuse tracing for PolicyArena simulations.

Provides a SimulationTracer that creates structured traces:

  Span (simulation run)
    ├─ Span (round 1)
    │   ├─ Generation (agent_A decide)
    │   ├─ Generation (agent_B decide)
    │   └─ Event (round results)
    ├─ Span (round 2)
    │   └─ ...
    └─ Event (simulation complete)

Usage:
    tracer = SimulationTracer.create(game_id="prisoners_dilemma", ...)
    tracer.start_round(1)
    gen = tracer.start_generation("agent_label", model="llama3", input=messages)
    gen.end(output="COOPERATE")
    tracer.end_round(results={...})
    tracer.finish()

If Langfuse credentials are not configured, all methods are no-ops.
"""

from __future__ import annotations

import os
from typing import Any


def _load_env() -> None:
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _langfuse_available() -> bool:
    """Check if Langfuse credentials are configured."""
    _load_env()
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    )


class _GenerationHandle:
    """Wraps a Langfuse generation for easy end() calls."""

    def __init__(self, generation: Any = None):
        self._gen = generation

    def end(
        self,
        output: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._gen is None:
            return
        update_kwargs: dict[str, Any] = {"output": output}
        if metadata:
            update_kwargs["metadata"] = metadata
        self._gen.update(**update_kwargs)
        self._gen.end()

    @property
    def is_active(self) -> bool:
        return self._gen is not None


class SimulationTracer:
    """Structured Langfuse tracing for a single simulation run.

    Call `SimulationTracer.create(...)` to get a tracer.
    If Langfuse is not configured, returns a no-op tracer.
    """

    def __init__(self, langfuse: Any, root_span: Any):
        self._langfuse = langfuse
        self._root_span = root_span
        self._current_round_span: Any = None
        self._active = True

    @classmethod
    def create(
        cls,
        *,
        game_id: str,
        agent_labels: list[str],
        agent_brains: list[str],
        n_rounds: int,
        seed: int | None = None,
        game_params: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> SimulationTracer:
        """Create a tracer for a simulation run.

        Returns a no-op tracer if Langfuse is not configured.
        """
        if not _langfuse_available():
            return _NoOpTracer()

        from langfuse import Langfuse

        langfuse = Langfuse()

        sim_metadata = {
            "game_id": game_id,
            "n_rounds": n_rounds,
            "seed": seed,
            "agents": [
                {"label": label, "brain": brain}
                for label, brain in zip(agent_labels, agent_brains, strict=False)
            ],
            "game_params": game_params or {},
        }

        # Create a root span for the entire simulation
        root_span = langfuse.start_span(
            name=f"simulation:{game_id}",
            metadata=sim_metadata,
        )

        # Set trace-level metadata
        root_span.update_trace(
            name=f"simulation:{game_id}",
            session_id=session_id,
            tags=["simulation", game_id],
            metadata=sim_metadata,
        )

        return cls(langfuse, root_span)

    @property
    def trace_id(self) -> str | None:
        if self._root_span is None:
            return None
        return self._root_span.trace_id

    def start_round(
        self, round_num: int, metadata: dict[str, Any] | None = None
    ) -> None:
        """Begin a new round span."""
        if not self._active:
            return
        self._current_round_span = self._root_span.start_span(
            name=f"round_{round_num}",
            metadata={"round": round_num, **(metadata or {})},
        )

    def end_round(self, results: dict[str, Any] | None = None) -> None:
        """End the current round span."""
        if not self._active or self._current_round_span is None:
            return
        if results:
            self._current_round_span.create_event(
                name="round_results",
                metadata=results,
            )
        self._current_round_span.end()
        self._current_round_span = None

    def start_generation(
        self,
        agent_label: str,
        *,
        model: str = "",
        input_messages: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> _GenerationHandle:
        """Start an LLM generation observation under the current round."""
        if not self._active:
            return _GenerationHandle(None)

        parent = self._current_round_span or self._root_span
        gen = parent.start_observation(
            as_type="generation",
            name=f"llm_decide:{agent_label}",
            model=model,
            input=input_messages,
            metadata={"agent": agent_label, **(metadata or {})},
        )
        return _GenerationHandle(gen)

    def log_event(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a discrete event under the current round or trace."""
        if not self._active:
            return
        parent = self._current_round_span or self._root_span
        parent.create_event(name=name, metadata=metadata or {})

    def finish(self, metadata: dict[str, Any] | None = None) -> None:
        """Finalize the trace and flush to Langfuse."""
        if not self._active:
            return
        if metadata:
            self._root_span.update(metadata=metadata)
        self._root_span.end()
        self._langfuse.flush()
        self._active = False


class _NoOpTracer(SimulationTracer):
    """No-op tracer when Langfuse is not configured."""

    def __init__(self) -> None:
        self._active = False

    @property
    def trace_id(self) -> str | None:
        return None

    def start_round(
        self, round_num: int, metadata: dict[str, Any] | None = None
    ) -> None:
        pass

    def end_round(self, results: dict[str, Any] | None = None) -> None:
        pass

    def start_generation(self, agent_label: str, **kwargs: Any) -> _GenerationHandle:
        return _GenerationHandle(None)

    def log_event(self, name: str, metadata: dict[str, Any] | None = None) -> None:
        pass

    def finish(self, metadata: dict[str, Any] | None = None) -> None:
        pass
