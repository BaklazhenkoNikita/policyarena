"""Ultimatum Game model.

Each round: agents are paired. In each pair, one proposes a split of the
stake, the other accepts or rejects. If accepted, both get their shares.
If rejected, both get 0.

In round-robin mode, each ordered pair plays once per round (A proposes to B,
B proposes to A — both directions).
"""

from __future__ import annotations

from itertools import permutations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.ultimatum.agents import UGAgent
from policy_arena.games.ultimatum.types import UGRoundResult
from policy_arena.metrics.entropy import shannon_entropy

OFFER_BINS = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-100%"]


def _bin_offer(offer: float, stake: float) -> str:
    if stake == 0:
        return OFFER_BINS[0]
    frac = offer / stake
    if frac < 0.1:
        return OFFER_BINS[0]
    elif frac < 0.2:
        return OFFER_BINS[1]
    elif frac < 0.3:
        return OFFER_BINS[2]
    elif frac < 0.4:
        return OFFER_BINS[3]
    elif frac < 0.5:
        return OFFER_BINS[4]
    elif frac < 0.6:
        return OFFER_BINS[5]
    else:
        return OFFER_BINS[6]


class UltimatumModel(mesa.Model):
    """Iterated Ultimatum Game with round-robin matching.

    Each step: every ordered pair (A,B) plays once — A proposes, B responds.
    NE: proposer offers minimum, responder accepts any > 0.
    """

    def __init__(
        self,
        brains: list[Brain],
        n_rounds: int = 100,
        stake: float = 100.0,
        labels: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.stake = stake

        self._round_offers: list[float] = []
        self._round_accepted: list[bool] = []
        self._round_total_payoff: float = 0.0
        self._round_max_payoff: float = 0.0

        for i, brain in enumerate(brains):
            label = labels[i] if labels else None
            UGAgent(self, brain=brain, label=label)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "cooperation_rate": lambda m: m._metric_acceptance_rate(),
                "nash_eq_distance": lambda m: m._metric_nash_distance(),
                "social_welfare": lambda m: m._metric_social_welfare(),
                "strategy_entropy": lambda m: m._metric_strategy_entropy(),
                "avg_offer_pct": lambda m: m._metric_avg_offer_pct(),
            },
            agent_reporters={
                "cumulative_payoff": "cumulative_payoff",
                "round_payoff": "round_payoff",
                "brain_name": "brain_name",
                "label": "label",
            },
        )

    def _metric_acceptance_rate(self) -> float:
        """Fraction of offers that were accepted."""
        if not self._round_accepted:
            return 0.0
        return sum(self._round_accepted) / len(self._round_accepted)

    def _metric_nash_distance(self) -> float:
        """NE = proposer offers minimum, responder accepts.

        Distance = fraction of interactions deviating from NE behavior.
        A deviation is: offer > 10% of stake OR rejection of positive offer.
        """
        if not self._round_offers:
            return 0.0
        deviations = 0
        for offer, accepted in zip(
            self._round_offers, self._round_accepted, strict=False
        ):
            offer_is_ne = offer <= self.stake * 0.1
            response_is_ne = accepted if offer > 0 else True
            if not (offer_is_ne and response_is_ne):
                deviations += 1
        return deviations / len(self._round_offers)

    def _metric_social_welfare(self) -> float:
        if self._round_max_payoff == 0:
            return 0.0
        return self._round_total_payoff / self._round_max_payoff

    def _metric_strategy_entropy(self) -> float:
        """Entropy over binned offer amounts."""
        if not self._round_offers:
            return 0.0
        bins = [_bin_offer(o, self.stake) for o in self._round_offers]
        return shannon_entropy(bins)

    def _metric_avg_offer_pct(self) -> float:
        if not self._round_offers or self.stake == 0:
            return 0.0
        return sum(o / self.stake for o in self._round_offers) / len(self._round_offers)

    def step(self) -> None:
        from collections import defaultdict

        from policy_arena.games.parallel import gather_decisions

        agents = list(self.agents)
        agent_map = {a.unique_id: a for a in agents}
        pairs = list(permutations(agents, 2))

        self._round_offers = []
        self._round_accepted = []
        self._round_total_payoff = 0.0
        self._round_max_payoff = self.stake * len(pairs)

        for agent in agents:
            agent.begin_round()

        # --- Phase 1: Batch all proposer decisions ---
        # Group pairs by proposer: proposer_id → [responder_id, ...]
        proposer_targets: dict[int, list[int]] = defaultdict(list)
        for proposer, responder in pairs:
            proposer_targets[proposer.unique_id].append(responder.unique_id)

        proposers_with_targets = [agent_map[pid] for pid in proposer_targets]

        def _propose(agent):
            targets = proposer_targets[agent.unique_id]
            return agent.make_proposals_batch(targets)

        max_w = getattr(self, "max_concurrent_llm", 1)
        all_proposals = gather_decisions(proposers_with_targets, _propose, max_w)

        # --- Phase 2: Batch all responder decisions ---
        # Group by responder: responder_id → [(proposer_id, offer), ...]
        responder_offers: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for proposer, responder in pairs:
            offer = all_proposals[proposer.unique_id][responder.unique_id]
            responder_offers[responder.unique_id].append((proposer.unique_id, offer))

        responders_with_offers = [agent_map[rid] for rid in responder_offers]

        def _respond(agent):
            offers = responder_offers[agent.unique_id]
            return agent.respond_to_offers_batch(offers)

        all_responses = gather_decisions(responders_with_offers, _respond, max_w)

        # --- Phase 3: Resolve payoffs ---
        for proposer, responder in pairs:
            offer = all_proposals[proposer.unique_id][responder.unique_id]
            accepted = all_responses[responder.unique_id][proposer.unique_id]

            if accepted:
                proposer_payoff = self.stake - offer
                responder_payoff = offer
            else:
                proposer_payoff = 0.0
                responder_payoff = 0.0

            proposer.record_result(
                UGRoundResult(
                    role="proposer",
                    offer=offer,
                    accepted=accepted,
                    payoff=proposer_payoff,
                    opponent_payoff=responder_payoff,
                    round_number=self.steps,
                ),
                opponent_id=responder.unique_id,
            )
            responder.record_result(
                UGRoundResult(
                    role="responder",
                    offer=offer,
                    accepted=accepted,
                    payoff=responder_payoff,
                    opponent_payoff=proposer_payoff,
                    round_number=self.steps,
                ),
                opponent_id=proposer.unique_id,
            )

            self._round_offers.append(offer)
            self._round_accepted.append(accepted)
            self._round_total_payoff += proposer_payoff + responder_payoff

        for agent in agents:
            agent.end_round()

        self.datacollector.collect(self)

        if self.steps >= self.n_rounds:
            self.running = False
