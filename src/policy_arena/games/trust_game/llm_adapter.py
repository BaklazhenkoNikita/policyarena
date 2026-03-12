"""Trust Game LLM adapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from policy_arena.brains.llm.dual_role_brain import DualRoleLLMBrain, RoleBrainConfig
from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona, _fmt_num


class TGInvestorDecision(BaseModel):
    """A single investor decision for one opponent."""

    opponent: str = Field(description="Name of the opponent")
    rationale: str = Field(description="1-2 sentence reasoning")
    investment: float = Field(description="Amount to invest (0 to endowment)")


class TGInvestorDecisionList(BaseModel):
    """Batch of investor decisions, one per opponent."""

    decisions: list[TGInvestorDecision]


class TGTrusteeDecision(BaseModel):
    """A single trustee decision for one investor."""

    opponent: str = Field(description="Name of the investor")
    rationale: str = Field(description="1-2 sentence reasoning")
    return_amount: float = Field(description="Amount to return (0 to amount received)")


class TGTrusteeDecisionList(BaseModel):
    """Batch of trustee decisions, one per investor."""

    decisions: list[TGTrusteeDecision]


TG_INVESTOR_PROMPT = """\
You are playing an Iterated Trust Game as an INVESTOR.

Each round you invest in multiple opponents simultaneously.
You have an endowment of {endowment}. You choose how much to send to each trustee.
Your investment is multiplied by {multiplier}x. The trustee then decides how much
to return to you.

Your payoff = endowment - investment + amount_returned
Trustee payoff = investment × {multiplier} - amount_returned

Nash Equilibrium: invest 0 (trustee would return 0 anyway by backward induction).
But if both cooperate, the surplus from multiplication benefits everyone.

{persona}

Provide one decision per opponent, in the same order presented.
Choose an investment amount between 0 and {endowment} for each."""

TG_TRUSTEE_PROMPT = """\
You are playing an Iterated Trust Game as a TRUSTEE.

Each round you receive investments from multiple investors simultaneously.
The investor sent some amount, which was multiplied by {multiplier}x.
You received the multiplied amount and decide how much to return.

Your payoff = amount_received - amount_returned
Investor payoff = endowment - investment + amount_returned

Nash Equilibrium: return 0 (maximize your payoff). But returning a fair
share builds trust and encourages future investment.

{persona}

Provide one decision per investor, in the same order presented.
Choose how much to return (0 to the amount you received) for each."""


def _tg_investor_observation_formatter(observations: list[Any]) -> str:
    if not observations:
        return ""

    round_num = observations[0].round_number + 1
    endowment = observations[0].endowment
    multiplier = observations[0].multiplier
    parts = [
        f"=== Round {round_num} — INVESTOR — {len(observations)} opponent(s) ===\n"
    ]
    parts.append(f"Endowment: {_fmt_num(endowment)}, Multiplier: {multiplier}x")

    for i, obs in enumerate(observations):
        opp_parts = [f"\n--- Opp {i + 1} ---"]

        if obs.opponent_past_returns:
            recent_ret = obs.opponent_past_returns[-5:]
            ret_str = ", ".join(f"{r:.1f}" for r in recent_ret)
            opp_parts.append(f"Their past returns to you: [{ret_str}]")

            if obs.my_past_investments:
                pairs = list(
                    zip(
                        obs.my_past_investments, obs.opponent_past_returns, strict=False
                    )
                )[-5:]
                roi_parts = []
                for inv, ret in pairs:
                    roi = (ret - inv) / inv * 100 if inv > 0 else 0
                    roi_parts.append(f"{roi:+.0f}%")
                opp_parts.append(f"ROI with this opp: [{', '.join(roi_parts)}]")
        else:
            opp_parts.append("No history yet with this opponent.")

        parts.append("\n".join(opp_parts))

    parts.append(f"\nChoose investment (0 to {_fmt_num(endowment)}) for each opponent.")
    return "\n\n".join(parts)


def _tg_trustee_observation_formatter(observations: list[Any]) -> str:
    if not observations:
        return ""

    round_num = observations[0].round_number + 1
    multiplier = observations[0].multiplier
    parts = [f"=== Round {round_num} — TRUSTEE — {len(observations)} investor(s) ===\n"]

    for i, obs in enumerate(observations):
        received = obs.amount_received or 0
        invested = received / multiplier if multiplier else 0
        opp_parts = [
            f"\n--- Opp {i + 1} — received: {_fmt_num(received)} "
            f"(invested {_fmt_num(invested)} × {multiplier}x) ---"
        ]

        if obs.opponent_past_investments:
            recent_inv = obs.opponent_past_investments[-5:]
            inv_str = ", ".join(f"{i:.1f}" for i in recent_inv)
            opp_parts.append(f"Their past investments: [{inv_str}]")

        parts.append("\n".join(opp_parts))

    parts.append("\nChoose return amount for each investor.")
    return "\n\n".join(parts)


def _tg_investor_action_extractor(
    response: TGInvestorDecisionList, n: int
) -> list[float]:
    actions: list[float] = []
    for d in response.decisions[:n]:
        actions.append(d.investment)
    while len(actions) < n:
        actions.append(5.0)
    return actions


def _tg_trustee_action_extractor(
    response: TGTrusteeDecisionList, n: int
) -> list[float]:
    actions: list[float] = []
    for d in response.decisions[:n]:
        actions.append(d.return_amount)
    while len(actions) < n:
        actions.append(5.0)
    return actions


def _tg_investor_result_formatter(result: Any) -> str:
    return ""


def _tg_trustee_result_formatter(result: Any) -> str:
    return ""


def _tg_investor_fallback(n: int) -> list[float]:
    return [5.0] * n


def _tg_trustee_fallback(n: int) -> list[float]:
    return [5.0] * n


class TGLLMBrain(DualRoleLLMBrain):
    """Combined LLM brain for the Trust Game — routes investor/trustee."""

    def __init__(
        self,
        *,
        provider="ollama",
        model="llama3",
        temperature=0.7,
        max_history=20,
        persona=None,
        characteristics=None,
        endowment=10.0,
        multiplier=3.0,
        api_key=None,
        base_url=None,
    ):
        persona_text = (
            persona
            if persona is not None
            else (_build_persona(characteristics) or DEFAULT_PERSONA)
        )

        super().__init__(
            provider=provider,
            model=model,
            temperature=temperature,
            max_history=max_history,
            api_key=api_key,
            base_url=base_url,
            role_configs=(
                RoleBrainConfig(
                    role_name="investor",
                    tag_prefix="[INVESTOR",
                    persona=TG_INVESTOR_PROMPT.format(
                        persona=persona_text,
                        endowment=_fmt_num(endowment),
                        multiplier=multiplier,
                    ),
                    output_schema=TGInvestorDecisionList,
                    batch_observation_formatter=_tg_investor_observation_formatter,
                    batch_action_extractor=_tg_investor_action_extractor,
                    result_formatter=_tg_investor_result_formatter,
                    fallback_action_factory=_tg_investor_fallback,
                ),
                RoleBrainConfig(
                    role_name="trustee",
                    tag_prefix="[TRUSTEE",
                    persona=TG_TRUSTEE_PROMPT.format(
                        persona=persona_text,
                        endowment=_fmt_num(endowment),
                        multiplier=multiplier,
                    ),
                    output_schema=TGTrusteeDecisionList,
                    batch_observation_formatter=_tg_trustee_observation_formatter,
                    batch_action_extractor=_tg_trustee_action_extractor,
                    result_formatter=_tg_trustee_result_formatter,
                    fallback_action_factory=_tg_trustee_fallback,
                ),
            ),
        )

    @property
    def _investor(self) -> LLMBrain:
        return self._brain_a

    @property
    def _trustee(self) -> LLMBrain:
        return self._brain_b


def tg_llm_combined(
    provider: str = "ollama",
    model: str = "llama3",
    temperature: float = 0.7,
    max_history: int = 20,
    persona: str | None = None,
    characteristics: dict[str, Any] | None = None,
    endowment: float = 10.0,
    multiplier: float = 3.0,
    api_key: str | None = None,
    base_url: str | None = None,
) -> TGLLMBrain:
    return TGLLMBrain(
        provider=provider,
        model=model,
        temperature=temperature,
        max_history=max_history,
        persona=persona,
        characteristics=characteristics,
        endowment=endowment,
        multiplier=multiplier,
        api_key=api_key,
        base_url=base_url,
    )
