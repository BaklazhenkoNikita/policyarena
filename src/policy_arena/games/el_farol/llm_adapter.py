"""El Farol Bar Problem LLM adapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona, _fmt_num


class EFDecision(BaseModel):
    """Decision for the El Farol Bar Problem."""
    rationale: str = Field(description="1-2 sentence reasoning")
    attend: bool = Field(description="true = go to the bar, false = stay home")


EF_SYSTEM_PROMPT = """\
You are playing the El Farol Bar Problem.

Each round, you independently decide whether to ATTEND the bar or STAY HOME.
There are {n_agents} agents total and the bar's comfort threshold is {threshold}.

Payoffs:
- Attend & fewer than {threshold} people go: +{attend_payoff} (good time!)
- Attend & {threshold} or more people go: {overcrowded_payoff} (overcrowded)
- Stay home: {stay_payoff}

The challenge: if everyone predicts low attendance and goes, it becomes crowded.
Your goal is to maximize your total payoff across all rounds.

{persona}

Decide whether to attend (true) or stay home (false)."""


def _ef_observation_formatter(observations: list[Any]) -> str:
    obs = observations[0]
    round_num = obs.round_number + 1
    parts = [f"=== Round {round_num} ===\n"]

    if obs.past_attendance:
        recent = obs.past_attendance[-10:]
        att_vs_thresh = ", ".join(
            f"{a}{'*' if a >= obs.threshold else ''}" for a in recent
        )
        parts.append(
            f"Recent attendance (* = overcrowded): [{att_vs_thresh}] "
            f"(threshold: {obs.threshold})"
        )

        avg = sum(recent) / len(recent)
        over = sum(1 for a in recent if a >= obs.threshold)
        parts.append(
            f"Avg attendance: {avg:.1f}, overcrowded {over}/{len(recent)} times"
        )

        # Moving average trend (compare first half vs second half of recent window)
        if len(recent) >= 4:
            mid = len(recent) // 2
            first_half_avg = sum(recent[:mid]) / mid
            second_half_avg = sum(recent[mid:]) / (len(recent) - mid)
            diff = second_half_avg - first_half_avg
            if diff > 1:
                trend = "trending UP"
            elif diff < -1:
                trend = "trending DOWN"
            else:
                trend = "stable"
            parts.append(
                f"Attendance trend: {trend} (early avg {first_half_avg:.1f} -> recent avg {second_half_avg:.1f})"
            )

        if obs.my_past_decisions:
            my_recent = obs.my_past_decisions[-10:]
            dec_str = ", ".join("Go" if d else "Stay" for d in my_recent)
            parts.append(f"Your recent decisions: [{dec_str}]")

            # Personal success rate (full history)
            times_attended = sum(1 for d in obs.my_past_decisions if d)
            if times_attended > 0:
                profitable_times = sum(
                    1
                    for d, a in zip(
                        obs.my_past_decisions, obs.past_attendance, strict=False
                    )
                    if d and a < obs.threshold
                )
                pct = profitable_times / times_attended * 100
                parts.append(
                    f"Personal success: attended {times_attended} times, "
                    f"profitable {profitable_times} times ({pct:.0f}%)"
                )

            if obs.my_past_payoffs:
                pay_recent = obs.my_past_payoffs[-10:]
                recent_total = sum(pay_recent)
                cumulative = sum(obs.my_past_payoffs)
                parts.append(
                    f"Payoffs: recent total={recent_total:.1f}, "
                    f"cumulative={cumulative:.1f}"
                )
        # Per-agent decisions (last 5 rounds)
        if obs.all_agent_decisions:
            recent_rounds = obs.all_agent_decisions[-5:]
            parts.append("--- Per-Agent Decisions (recent rounds) ---")
            agents = list(recent_rounds[0].keys())
            header = "Round  " + "  ".join(f"{a[:10]:>10}" for a in agents)
            parts.append(header)
            start_round = max(1, len(obs.all_agent_decisions) - 4)
            for i, rd in enumerate(recent_rounds):
                rnum = start_round + i
                vals = "  ".join(
                    f"{'Go':>10}" if rd.get(a) else f"{'Stay':>10}" for a in agents
                )
                parts.append(f"  {rnum:<5}{vals}")

    else:
        parts.append("No history yet — first round.")

    parts.append("\nDecide: attend or stay home?")
    return "\n\n".join(parts)


def _ef_action_extractor(response: EFDecision, n: int) -> list[bool]:
    return [response.attend]


def _ef_result_formatter(result: Any) -> str:
    action = "attended" if result.attended else "stayed home"
    crowded = "overcrowded" if result.attendance >= result.threshold else "not crowded"
    return (
        f"[Result: you {action}, attendance was {result.attendance}/{result.threshold} "
        f"({crowded}), payoff: {result.payoff}]"
    )


def _ef_fallback(n: int) -> list[bool]:
    return [True] * n


def ef_llm(
    provider: str = "ollama",
    model: str = "llama3",
    temperature: float = 0.7,
    max_history: int = 20,
    persona: str | None = None,
    characteristics: dict[str, Any] | None = None,
    n_agents: int = 10,
    threshold: int = 6,
    attend_payoff: float = 1.0,
    overcrowded_payoff: float = -1.0,
    stay_payoff: float = 0.0,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMBrain:
    """Create an LLM brain configured for the El Farol Bar Problem."""
    persona_text = (
        persona
        if persona is not None
        else (_build_persona(characteristics) or DEFAULT_PERSONA)
    )

    system_prompt = EF_SYSTEM_PROMPT.format(
        persona=persona_text,
        n_agents=n_agents,
        threshold=threshold,
        attend_payoff=_fmt_num(attend_payoff),
        overcrowded_payoff=_fmt_num(overcrowded_payoff),
        stay_payoff=_fmt_num(stay_payoff),
    )

    return LLMBrain(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        persona=system_prompt,
        output_schema=EFDecision,
        batch_observation_formatter=_ef_observation_formatter,
        batch_action_extractor=_ef_action_extractor,
        result_formatter=_ef_result_formatter,
        fallback_action_factory=_ef_fallback,
        temperature=temperature,
        max_history=max_history,
        brain_name=f"llm({provider}/{model})",
    )
