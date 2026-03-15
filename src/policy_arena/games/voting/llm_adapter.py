"""Voting & Election Game LLM adapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona, _fmt_num


class VotingDecision(BaseModel):
    """Decision for the Voting & Election Game."""

    rationale: str = Field(description="1-2 sentence reasoning")
    vote: list[int] = Field(
        description="Your vote: for plurality a single-element list [candidate_index], "
        "for approval a list of approved candidate indices, "
        "for borda a full ranking of candidate indices (best first)"
    )


VOTING_SYSTEM_PROMPT = """\
You are a voter in an Iterated Voting & Election Game.

There are {n_candidates} candidates positioned on a 1D issue space (0-100).
Candidate positions: {candidate_positions}

Your ideal point: {ideal_point}

Voting rule: {voting_rule}
{rule_instructions}

Your payoff each round = -abs(winner_position - your_ideal_point).
Closer winners give higher payoffs (less negative). Best possible payoff is 0.

Your goal is to maximize your total payoff across all rounds by voting
strategically to elect candidates close to your ideal point.

{persona}

{vote_format}"""

RULE_INSTRUCTIONS = {
    "plurality": "Vote for exactly ONE candidate. The candidate with the most votes wins.",
    "approval": "Vote for any SUBSET of candidates you approve. The candidate with the most approvals wins.",
    "borda": "Rank ALL candidates from best to worst. Points: {m_minus_1} for 1st, {m_minus_2} for 2nd, ..., 0 for last. Most points wins.",
}

VOTE_FORMAT = {
    "plurality": "Respond with vote as a single-element list, e.g. [2] to vote for candidate 2.",
    "approval": "Respond with vote as a list of approved candidate indices, e.g. [0, 2, 3].",
    "borda": "Respond with vote as a list ranking all candidates best-to-worst, e.g. [2, 0, 3, 1].",
}


def _voting_observation_formatter(observations: list[Any]) -> str:
    obs = observations[0]
    round_num = obs.round_number + 1
    parts = [f"=== Round {round_num} ===\n"]

    pos_str = ", ".join(
        f"C{i}: {_fmt_num(p)}" for i, p in enumerate(obs.candidate_positions)
    )
    parts.append(f"Candidates: {pos_str}")
    parts.append(f"Your ideal point: {_fmt_num(obs.my_ideal_point)}")
    parts.append(f"Voting rule: {obs.voting_rule}")

    if obs.past_winners:
        recent_winners = obs.past_winners[-10:]
        winner_str = ", ".join(f"C{w}" for w in recent_winners)
        parts.append(f"Recent winners: [{winner_str}]")

        recent_pos = obs.past_winner_positions[-10:]
        pos_str = ", ".join(f"{_fmt_num(p)}" for p in recent_pos)
        parts.append(f"Winner positions: [{pos_str}]")

        if obs.my_past_payoffs:
            recent_payoffs = obs.my_past_payoffs[-10:]
            total = sum(recent_payoffs)
            avg = total / len(recent_payoffs)
            cumulative = sum(obs.my_past_payoffs)
            parts.append(f"Your recent payoffs: avg={avg:.2f}")
            parts.append(f"Your cumulative payoff: {cumulative:.1f}")

        if obs.all_vote_counts:
            recent_counts = obs.all_vote_counts[-5:]
            parts.append("--- Vote Counts (recent rounds) ---")
            header = "Round  " + "  ".join(f"C{i:>3}" for i in range(obs.n_candidates))
            parts.append(header)
            start_round = max(1, len(obs.all_vote_counts) - 4)
            for i, counts in enumerate(recent_counts):
                rnum = start_round + i
                vals = "  ".join(
                    f"{counts.get(c, 0):>4}" for c in range(obs.n_candidates)
                )
                parts.append(f"  {rnum:<5}{vals}")

        if obs.my_past_votes:
            recent_votes = obs.my_past_votes[-5:]
            vote_str = ", ".join(str(v) for v in recent_votes)
            parts.append(f"Your recent votes: [{vote_str}]")
    else:
        parts.append("No history yet -- first round.")

    parts.append(f"\nCast your vote (rule: {obs.voting_rule}).")
    return "\n\n".join(parts)


def _voting_action_extractor(response: VotingDecision, n: int) -> list[Any]:
    return [response.vote]


def _voting_result_formatter(result: Any) -> str:
    return (
        f"[Result: you voted {result.my_vote}, "
        f"winner was C{result.winner_id} (position {_fmt_num(result.winner_position)}), "
        f"vote counts: {result.vote_counts}, "
        f"your payoff: {result.payoff:.1f}]"
    )


def _voting_fallback(n: int) -> list[Any]:
    return [[0]] * n  # vote for candidate 0


def voting_llm(
    provider: str = "ollama",
    model: str = "llama3",
    temperature: float = 0.7,
    max_history: int = 20,
    persona: str | None = None,
    characteristics: dict[str, Any] | None = None,
    n_candidates: int = 4,
    candidate_positions: list[float] | None = None,
    voting_rule: str = "plurality",
    ideal_point: float = 50.0,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMBrain:
    """Create an LLM brain configured for the Voting & Election Game."""
    persona_text = (
        persona
        if persona is not None
        else (_build_persona(characteristics) or DEFAULT_PERSONA)
    )

    if candidate_positions is None:
        if n_candidates == 1:
            candidate_positions = [50.0]
        else:
            candidate_positions = [
                i * 100.0 / (n_candidates - 1) for i in range(n_candidates)
            ]

    pos_str = ", ".join(
        f"C{i}: {_fmt_num(p)}" for i, p in enumerate(candidate_positions)
    )

    rule_instr = RULE_INSTRUCTIONS.get(voting_rule, RULE_INSTRUCTIONS["plurality"])
    if voting_rule == "borda":
        rule_instr = rule_instr.format(
            m_minus_1=n_candidates - 1, m_minus_2=max(0, n_candidates - 2)
        )

    vote_fmt = VOTE_FORMAT.get(voting_rule, VOTE_FORMAT["plurality"])

    system_prompt = VOTING_SYSTEM_PROMPT.format(
        persona=persona_text,
        n_candidates=n_candidates,
        candidate_positions=pos_str,
        ideal_point=_fmt_num(ideal_point),
        voting_rule=voting_rule,
        rule_instructions=rule_instr,
        vote_format=vote_fmt,
    )

    return LLMBrain(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        persona=system_prompt,
        output_schema=VotingDecision,
        batch_observation_formatter=_voting_observation_formatter,
        batch_action_extractor=_voting_action_extractor,
        result_formatter=_voting_result_formatter,
        fallback_action_factory=_voting_fallback,
        temperature=temperature,
        max_history=max_history,
        brain_name=f"llm({provider}/{model})",
    )
