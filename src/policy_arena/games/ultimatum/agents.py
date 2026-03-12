"""Ultimatum Game agent."""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.ultimatum.types import UGObservation, UGRoundResult


class UGAgent(mesa.Agent):
    """Agent in an Ultimatum Game tournament.

    Plays both proposer and responder roles across rounds.
    """

    def __init__(self, model: mesa.Model, brain: Brain, label: str | None = None):
        super().__init__(model)
        self.brain = brain
        self.label = label or f"{brain.name}_{self.unique_id}"
        self.cumulative_payoff: float = 0.0
        self.round_payoff: float = 0.0

        self._offers_made: list[float] = []
        self._offers_received: list[float] = []
        self._responses_given: list[bool] = []
        self._opponent_offers: dict[int, list[float]] = {}

        self._last_role: str | None = None
        self._last_offer: float | None = None
        self._last_accepted: bool | None = None
        self._round_payoff_accum: float = 0.0

        # Per-round tracking for consolidated summaries
        self._round_proposer_results: list[tuple[float, bool, float]] = []
        self._round_responder_results: list[tuple[float, bool, float]] = []
        self._round_proposer_payoff: float = 0.0
        self._round_responder_payoff: float = 0.0

    @property
    def brain_name(self) -> str:
        return self.brain.name

    def begin_round(self) -> None:
        """Reset per-round accumulators. Called by the model at the start of each step."""
        self._round_payoff_accum = 0.0
        self._round_proposer_results = []
        self._round_responder_results = []
        self._round_proposer_payoff = 0.0
        self._round_responder_payoff = 0.0

    def make_proposal(self, opponent_id: int) -> float:
        obs = self._proposer_observation(opponent_id)
        raw = self.brain.decide(obs)
        return max(0.0, min(self.model.stake, float(raw)))

    def make_proposals_batch(
        self, opponent_ids: list[int]
    ) -> dict[int, float]:
        """Batch-decide proposals for all opponents in a single LLM call."""
        observations = [self._proposer_observation(oid) for oid in opponent_ids]
        raw_actions = self.brain.decide_batch(observations)
        return {
            oid: max(0.0, min(self.model.stake, float(raw)))
            for oid, raw in zip(opponent_ids, raw_actions, strict=False)
        }

    def respond_to_offers_batch(
        self, offers: list[tuple[int, float]]
    ) -> dict[int, bool]:
        """Batch-decide responses for all received offers in a single LLM call."""
        observations = [self._responder_observation(oid, offer) for oid, offer in offers]
        raw_actions = self.brain.decide_batch(observations)
        return {
            oid: bool(raw)
            for (oid, _), raw in zip(offers, raw_actions, strict=False)
        }

    def respond_to_offer(self, offer: float, opponent_id: int) -> bool:
        obs = self._responder_observation(opponent_id, offer)
        return bool(self.brain.decide(obs))

    def _proposer_observation(self, opponent_id: int) -> UGObservation:
        return UGObservation(
            role="proposer",
            stake=self.model.stake,
            round_number=self.model.steps,
            my_past_offers_made=list(self._offers_made),
            opponent_past_offers=self._opponent_offers.get(opponent_id, []),
        )

    def _responder_observation(
        self, opponent_id: int, offer: float
    ) -> UGObservation:
        return UGObservation(
            role="responder",
            stake=self.model.stake,
            offer=offer,
            round_number=self.model.steps,
            my_past_offers_received=list(self._offers_received),
            my_past_responses=list(self._responses_given),
            opponent_past_offers=self._opponent_offers.get(opponent_id, []),
        )

    def record_result(self, result: UGRoundResult, opponent_id: int) -> None:
        self.cumulative_payoff += result.payoff
        self._round_payoff_accum += result.payoff
        self.round_payoff = self._round_payoff_accum

        self._last_role = result.role
        self._last_offer = result.offer
        self._last_accepted = result.accepted

        if result.role == "proposer":
            self._offers_made.append(result.offer)
            self._opponent_offers.setdefault(opponent_id, []).append(result.offer)
            self._round_proposer_results.append(
                (result.offer, result.accepted, result.payoff)
            )
            self._round_proposer_payoff += result.payoff
        else:
            self._offers_received.append(result.offer)
            self._responses_given.append(result.accepted)
            self._round_responder_results.append(
                (result.offer, result.accepted, result.payoff)
            )
            self._round_responder_payoff += result.payoff
        self.brain.update(result)

    def end_round(self) -> None:
        """Send consolidated round summaries to the brain."""
        step = self.model.steps

        if self._round_proposer_results:
            parts = [
                f"[PROPOSER Round {step} — total proposer payoff: "
                f"{self._round_proposer_payoff:.1f}]"
            ]
            for idx, (offer, accepted, payoff) in enumerate(
                self._round_proposer_results
            ):
                status = "accepted" if accepted else "REJECTED"
                parts.append(
                    f"  Opp {idx + 1}: offered {offer:.0f}, {status}, "
                    f"payoff={payoff:.1f}"
                )
            self.brain.update_round_summary("\n".join(parts))

        if self._round_responder_results:
            parts = [
                f"[RESPONDER Round {step} — total responder payoff: "
                f"{self._round_responder_payoff:.1f}]"
            ]
            for idx, (offer, accepted, payoff) in enumerate(
                self._round_responder_results
            ):
                action = "accepted" if accepted else "rejected"
                pct = offer / self.model.stake * 100
                parts.append(
                    f"  From opp {idx + 1}: offer={offer:.0f} ({pct:.0f}%), "
                    f"you {action}, payoff={payoff:.1f}"
                )
            self.brain.update_round_summary("\n".join(parts))
