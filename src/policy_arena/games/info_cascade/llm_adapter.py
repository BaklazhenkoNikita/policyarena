"""Information Cascade LLM adapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona


class CascadeDecision(BaseModel):
    """Decision for the Information Cascade game."""

    rationale: str = Field(description="1-2 sentence reasoning")
    choice: str = Field(description="Your choice: 'A' or 'B'")


CASCADE_SYSTEM_PROMPT = """\
You are playing an Information Cascade game.

Each round, a true state (A or B) is drawn with equal probability. You receive
a private signal that is correct with probability {signal_accuracy}. You also
observe the choices made by agents who decided before you (but NOT their signals).

You must choose A or B. You earn +1 if your choice matches the true state, -1 otherwise.

Key insight: When many agents before you have chosen the same option, it may be
rational to follow the herd even if your private signal disagrees — this is an
information cascade. However, cascades can form on the WRONG answer, especially
if early agents happened to get incorrect signals.

Bayesian reasoning: Your signal gives you log-likelihood evidence toward one state.
Each prior agent's choice also provides evidence (assuming they are rational).
A rational agent weighs all evidence and chooses the more likely state.

You are agent #{my_position} of {n_players} (0-indexed). {window_info}

Your goal is to maximize your total payoff across all rounds.

{persona}

Choose A or B."""


def _cascade_observation_formatter(observations: list[Any]) -> str:
    obs = observations[0]
    round_num = obs.round_number + 1
    parts = [f"=== Round {round_num} ===\n"]
    parts.append(f"Signal accuracy: {obs.signal_accuracy}")
    parts.append(f"Your position: {obs.my_position} of {obs.n_players}")
    parts.append(f"Your private signal: {obs.my_signal}")

    if obs.prior_choices:
        choices_str = ", ".join(obs.prior_choices)
        count_a = obs.prior_choices.count("A")
        count_b = obs.prior_choices.count("B")
        parts.append(f"Prior choices: [{choices_str}] (A: {count_a}, B: {count_b})")
    else:
        parts.append("You are first to decide — no prior choices to observe.")

    if obs.my_past_choices:
        recent_choices = obs.my_past_choices[-10:]
        recent_payoffs = obs.my_past_payoffs[-10:]
        recent_states = obs.past_true_states[-10:]

        choices_str = ", ".join(recent_choices)
        payoffs_str = ", ".join(f"{p:+.0f}" for p in recent_payoffs)
        states_str = ", ".join(recent_states)

        parts.append(f"\nYour recent choices:  [{choices_str}]")
        parts.append(f"True states were:     [{states_str}]")
        parts.append(f"Your recent payoffs:  [{payoffs_str}]")

        correct = sum(
            1 for c, s in zip(recent_choices, recent_states, strict=False) if c == s
        )
        parts.append(f"Recent accuracy: {correct}/{len(recent_choices)}")

        cumulative = sum(obs.my_past_payoffs)
        parts.append(f"Cumulative payoff: {cumulative:+.0f}")

        # Show recent cascade patterns
        if obs.past_round_choices:
            parts.append("\n--- Recent Round Outcomes ---")
            recent_rounds = obs.past_round_choices[-5:]
            recent_states_subset = obs.past_true_states[-5:]
            start_round = max(1, len(obs.past_round_choices) - 4)
            for j, (rd_choices, state) in enumerate(
                zip(recent_rounds, recent_states_subset, strict=False)
            ):
                rnum = start_round + j
                all_str = ", ".join(rd_choices)
                correct_count = sum(1 for c in rd_choices if c == state)
                parts.append(
                    f"  Round {rnum}: [{all_str}] | True: {state} | "
                    f"Correct: {correct_count}/{len(rd_choices)}"
                )
    else:
        parts.append("\nNo history yet — first round.")

    parts.append("\nChoose A or B.")
    return "\n\n".join(parts)


def _cascade_action_extractor(response: CascadeDecision, n: int) -> list[str]:
    choice = response.choice.strip().upper()
    if choice not in ("A", "B"):
        choice = "A"
    return [choice]


def _cascade_result_formatter(result: Any) -> str:
    return (
        f"[Result: you chose {result.my_choice}, signal was {result.my_signal}, "
        f"true state was {result.true_state}, "
        f"payoff: {result.payoff:+.0f}, "
        f"cascade: {'yes' if result.was_cascade else 'no'}]"
    )


def _cascade_fallback(n: int) -> list[str]:
    return ["A"] * n


def cascade_llm(
    provider: str = "ollama",
    model: str = "llama3",
    temperature: float = 0.7,
    max_history: int = 20,
    persona: str | None = None,
    characteristics: dict[str, Any] | None = None,
    signal_accuracy: float = 0.7,
    n_players: int = 5,
    my_position: int = 0,
    observation_window: int = 0,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMBrain:
    """Create an LLM brain configured for the Information Cascade game."""
    persona_text = (
        persona
        if persona is not None
        else (_build_persona(characteristics) or DEFAULT_PERSONA)
    )

    window_info = (
        f"You can see the last {observation_window} choices before you."
        if observation_window > 0
        else "You can see all choices made before you."
    )

    system_prompt = CASCADE_SYSTEM_PROMPT.format(
        persona=persona_text,
        signal_accuracy=signal_accuracy,
        n_players=n_players,
        my_position=my_position,
        window_info=window_info,
    )

    return LLMBrain(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        persona=system_prompt,
        output_schema=CascadeDecision,
        batch_observation_formatter=_cascade_observation_formatter,
        batch_action_extractor=_cascade_action_extractor,
        result_formatter=_cascade_result_formatter,
        fallback_action_factory=_cascade_fallback,
        temperature=temperature,
        max_history=max_history,
        brain_name=f"llm({provider}/{model})",
    )
