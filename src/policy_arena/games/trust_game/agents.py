"""Trust Game agent."""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.trust_game.types import TGObservation, TGRoundResult


class TGAgent(mesa.Agent):
    """Agent in a Trust Game tournament.

    Plays both investor and trustee roles across rounds.
    """

    def __init__(self, model: mesa.Model, brain: Brain, label: str | None = None):
        super().__init__(model)
        self.brain = brain
        self.label = label or f"{brain.name}_{self.unique_id}"
        self.cumulative_payoff: float = 0.0
        self.round_payoff: float = 0.0

        self._investments_made: list[float] = []
        self._returns_made: list[float] = []
        self._opponent_investments: dict[int, list[float]] = {}
        self._opponent_returns: dict[int, list[float]] = {}

        self._last_role: str | None = None
        self._last_investment: float | None = None
        self._last_return: float | None = None
        self._round_payoff_accum: float = 0.0

        # Per-round tracking for consolidated summaries
        self._round_investor_results: list[tuple[float, float, float]] = []
        self._round_trustee_results: list[tuple[float, float, float, float]] = []
        self._round_investor_payoff: float = 0.0
        self._round_trustee_payoff: float = 0.0

    @property
    def brain_name(self) -> str:
        return self.brain.name

    def begin_round(self) -> None:
        self._round_payoff_accum = 0.0
        self._round_investor_results = []
        self._round_trustee_results = []
        self._round_investor_payoff = 0.0
        self._round_trustee_payoff = 0.0

    def make_investment(self, opponent_id: int) -> float:
        obs = self._investor_observation(opponent_id)
        raw = self.brain.decide(obs)
        return max(0.0, min(self.model.endowment, float(raw)))

    def make_investments_batch(
        self, opponent_ids: list[int]
    ) -> dict[int, float]:
        """Batch-decide investments for all opponents in a single LLM call."""
        observations = [self._investor_observation(oid) for oid in opponent_ids]
        raw_actions = self.brain.decide_batch(observations)
        return {
            oid: max(0.0, min(self.model.endowment, float(raw)))
            for oid, raw in zip(opponent_ids, raw_actions, strict=False)
        }

    def decide_return(self, amount_received: float, opponent_id: int) -> float:
        obs = self._trustee_observation(opponent_id, amount_received)
        raw = self.brain.decide(obs)
        return max(0.0, min(amount_received, float(raw)))

    def decide_returns_batch(
        self, received: list[tuple[int, float]]
    ) -> dict[int, float]:
        """Batch-decide returns for all investors in a single LLM call."""
        observations = [
            self._trustee_observation(oid, amt) for oid, amt in received
        ]
        raw_actions = self.brain.decide_batch(observations)
        return {
            oid: max(0.0, min(amt, float(raw)))
            for (oid, amt), raw in zip(received, raw_actions, strict=False)
        }

    def _investor_observation(self, opponent_id: int) -> TGObservation:
        return TGObservation(
            role="investor",
            endowment=self.model.endowment,
            multiplier=self.model.multiplier,
            round_number=self.model.steps,
            my_past_investments=list(self._investments_made),
            opponent_past_returns=self._opponent_returns.get(opponent_id, []),
        )

    def _trustee_observation(
        self, opponent_id: int, amount_received: float
    ) -> TGObservation:
        return TGObservation(
            role="trustee",
            endowment=self.model.endowment,
            multiplier=self.model.multiplier,
            amount_received=amount_received,
            round_number=self.model.steps,
            my_past_returns=list(self._returns_made),
            opponent_past_investments=self._opponent_investments.get(
                opponent_id, []
            ),
        )

    def record_result(self, result: TGRoundResult, opponent_id: int) -> None:
        self.cumulative_payoff += result.payoff
        self._round_payoff_accum += result.payoff
        self.round_payoff = self._round_payoff_accum

        self._last_role = result.role
        self._last_investment = result.investment
        self._last_return = result.amount_returned

        if result.role == "investor":
            self._investments_made.append(result.investment)
            self._opponent_returns.setdefault(opponent_id, []).append(
                result.amount_returned
            )
            self._round_investor_results.append(
                (result.investment, result.amount_returned, result.payoff)
            )
            self._round_investor_payoff += result.payoff
        else:
            self._returns_made.append(result.amount_returned)
            self._opponent_investments.setdefault(opponent_id, []).append(
                result.investment
            )
            self._round_trustee_results.append(
                (result.investment, result.amount_received, result.amount_returned, result.payoff)
            )
            self._round_trustee_payoff += result.payoff
        self.brain.update(result)

    def end_round(self) -> None:
        """Send consolidated round summaries to the brain."""
        step = self.model.steps

        if self._round_investor_results:
            parts = [
                f"[INVESTOR Round {step} — total investor payoff: "
                f"{self._round_investor_payoff:.1f}]"
            ]
            for idx, (inv, ret, payoff) in enumerate(
                self._round_investor_results
            ):
                roi = (ret - inv) / inv * 100 if inv > 0 else 0
                parts.append(
                    f"  Opp {idx + 1}: invested {inv:.1f}, returned {ret:.1f}, "
                    f"ROI={roi:+.0f}%, payoff={payoff:.1f}"
                )
            self.brain.update_round_summary("\n".join(parts))

        if self._round_trustee_results:
            parts = [
                f"[TRUSTEE Round {step} — total trustee payoff: "
                f"{self._round_trustee_payoff:.1f}]"
            ]
            for idx, (inv, received, returned, payoff) in enumerate(
                self._round_trustee_results
            ):
                ret_pct = returned / received * 100 if received > 0 else 0
                parts.append(
                    f"  From opp {idx + 1}: received {received:.1f} "
                    f"(invested {inv:.1f}), returned {returned:.1f} "
                    f"({ret_pct:.0f}%), payoff={payoff:.1f}"
                )
            self.brain.update_round_summary("\n".join(parts))
