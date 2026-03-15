"""Voting & Election Game model.

N voters, M candidates positioned in a 1D issue space. Each voter has an
ideal point. Voters cast votes under a specified voting rule (plurality,
approval, or borda). The winning candidate is determined by the rule.

Payoff = -abs(winner_position - my_ideal_point).
"""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.voting.agents import VotingAgent
from policy_arena.games.voting.types import VotingRoundResult
from policy_arena.metrics.entropy import normalized_shannon_entropy
from policy_arena.metrics.social_welfare import compute_social_welfare


class VotingModel(mesa.Model):
    """Voting & Election Game.

    Each step: all agents simultaneously vote, winner is determined by
    the voting rule, payoffs are computed based on distance to winner.
    """

    def __init__(
        self,
        brains: list[Brain],
        n_rounds: int = 100,
        n_candidates: int = 4,
        candidate_positions: list[float] | None = None,
        voting_rule: str = "plurality",
        ideal_points: list[float] | None = None,
        labels: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.n_candidates = n_candidates
        self.voting_rule = voting_rule

        if candidate_positions is not None:
            self.candidate_positions = list(candidate_positions)
        else:
            # Evenly spaced from 0 to 100
            if n_candidates == 1:
                self.candidate_positions = [50.0]
            else:
                self.candidate_positions = [
                    i * 100.0 / (n_candidates - 1) for i in range(n_candidates)
                ]

        self.winner_history: list[int] = []
        self.winner_position_history: list[float] = []
        self.vote_count_history: list[dict[int, int]] = []

        self._round_total_payoff: float = 0.0
        self._round_max_payoff: float = 0.0
        self._round_votes: list[int] = []  # for plurality entropy
        self._round_sincere_count: int = 0

        for i, brain in enumerate(brains):
            label = labels[i] if labels else None
            if ideal_points is not None:
                ip = ideal_points[i]
            else:
                ip = self.random.uniform(0, 100)
            VotingAgent(self, brain=brain, ideal_point=ip, label=label)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "winner_position": lambda m: m._metric_winner_position(),
                "sincere_voting_rate": lambda m: m._metric_sincere_voting_rate(),
                "effective_number_of_candidates": lambda m: (
                    m._metric_effective_candidates()
                ),
                "social_welfare": lambda m: compute_social_welfare(m),
                "strategy_entropy": lambda m: m._metric_strategy_entropy(),
            },
            agent_reporters={
                "cumulative_payoff": "cumulative_payoff",
                "round_payoff": "round_payoff",
                "last_vote": "last_vote",
                "brain_name": "brain_name",
                "label": "label",
            },
        )

    def _closest_candidate(self, ideal_point: float) -> int:
        """Return the index of the candidate closest to the ideal point."""
        return min(
            range(self.n_candidates),
            key=lambda c: abs(self.candidate_positions[c] - ideal_point),
        )

    def _tally_plurality(self, votes: dict[int, int]) -> dict[int, int]:
        """Tally plurality votes. Each vote is a single candidate index."""
        counts: dict[int, int] = {c: 0 for c in range(self.n_candidates)}
        for vote in votes.values():
            v = int(vote)
            if 0 <= v < self.n_candidates:
                counts[v] = counts.get(v, 0) + 1
        return counts

    def _tally_approval(self, votes: dict[int, list[int]]) -> dict[int, int]:
        """Tally approval votes. Each vote is a list of approved candidate indices."""
        counts: dict[int, int] = {c: 0 for c in range(self.n_candidates)}
        for vote in votes.values():
            if isinstance(vote, (list, tuple)):
                for v in vote:
                    v = int(v)
                    if 0 <= v < self.n_candidates:
                        counts[v] = counts.get(v, 0) + 1
            else:
                # Fallback: treat as single approval
                v = int(vote)
                if 0 <= v < self.n_candidates:
                    counts[v] = counts.get(v, 0) + 1
        return counts

    def _tally_borda(self, votes: dict[int, list[int]]) -> dict[int, int]:
        """Tally Borda votes. Each vote is a ranking (list of candidate indices).

        Points: M-1 for 1st place, M-2 for 2nd, ..., 0 for last.
        """
        m = self.n_candidates
        counts: dict[int, int] = {c: 0 for c in range(m)}
        for vote in votes.values():
            if isinstance(vote, (list, tuple)):
                for rank, candidate in enumerate(vote):
                    candidate = int(candidate)
                    if 0 <= candidate < m:
                        counts[candidate] = counts.get(candidate, 0) + (m - 1 - rank)
            else:
                # Fallback: treat as single top candidate getting max points
                v = int(vote)
                if 0 <= v < m:
                    counts[v] = counts.get(v, 0) + (m - 1)
        return counts

    def _determine_winner(self, counts: dict[int, int]) -> int:
        """Return the candidate with the most votes/points. Ties broken by lowest index."""
        return max(counts, key=lambda c: (counts[c], -c))

    def _metric_winner_position(self) -> float:
        if self.winner_position_history:
            return self.winner_position_history[-1]
        return 0.0

    def _metric_sincere_voting_rate(self) -> float:
        agents = list(self.agents)
        if not agents:
            return 0.0
        return self._round_sincere_count / len(agents)

    def _metric_effective_candidates(self) -> float:
        """Effective number of candidates: 1 / sum(vote_share^2)."""
        if not self.vote_count_history:
            return 0.0
        counts = self.vote_count_history[-1]
        total = sum(counts.values())
        if total == 0:
            return 0.0
        sum_sq = sum((v / total) ** 2 for v in counts.values() if v > 0)
        if sum_sq == 0:
            return 0.0
        return 1.0 / sum_sq

    def _metric_strategy_entropy(self) -> float:
        """Shannon entropy over vote choices."""
        if not self._round_votes:
            return 0.0
        return normalized_shannon_entropy(
            self._round_votes, n_categories=self.n_candidates
        )

    def step(self) -> None:
        agents = list(self.agents)
        n = len(agents)

        from policy_arena.games.parallel import gather_decisions

        max_w = getattr(self, "max_concurrent_llm", 1)
        raw_votes = gather_decisions(agents, lambda a: a.decide(), max_w)

        # Tally votes based on rule
        if self.voting_rule == "approval":
            counts = self._tally_approval(raw_votes)
        elif self.voting_rule == "borda":
            counts = self._tally_borda(raw_votes)
        else:  # plurality
            counts = self._tally_plurality(raw_votes)

        winner_id = self._determine_winner(counts)
        winner_position = self.candidate_positions[winner_id]

        # Track sincere voting
        sincere_count = 0
        for agent in agents:
            closest = self._closest_candidate(agent.ideal_point)
            vote = raw_votes[agent.unique_id]
            if self.voting_rule == "plurality":
                if int(vote) == closest:
                    sincere_count += 1
            elif self.voting_rule == "approval":
                if isinstance(vote, (list, tuple)) and closest in vote:
                    sincere_count += 1
            elif (
                self.voting_rule == "borda"
                and isinstance(vote, (list, tuple))
                and len(vote) > 0
                and int(vote[0]) == closest
            ):
                sincere_count += 1
        self._round_sincere_count = sincere_count

        # For entropy: flatten votes to candidate indices
        if self.voting_rule == "plurality":
            self._round_votes = [int(raw_votes[a.unique_id]) for a in agents]
        elif self.voting_rule == "approval":
            # Use the first approved candidate for entropy
            flat = []
            for a in agents:
                v = raw_votes[a.unique_id]
                if isinstance(v, (list, tuple)) and v:
                    flat.append(int(v[0]))
                else:
                    flat.append(int(v))
            self._round_votes = flat
        else:  # borda
            # Use top-ranked candidate for entropy
            flat = []
            for a in agents:
                v = raw_votes[a.unique_id]
                if isinstance(v, (list, tuple)) and v:
                    flat.append(int(v[0]))
                else:
                    flat.append(int(v))
            self._round_votes = flat

        # Compute payoffs and record results
        self._round_total_payoff = 0.0
        # Max payoff: all voters get payoff 0 (winner at their ideal point)
        # Since payoffs are negative, max is 0. We use absolute total for welfare.
        # Social welfare = total_payoff / max_payoff; with negatives, define max as 0
        # and guard against division. Use: welfare = 1 + (total_payoff / worst_case)
        # where worst_case = -100 * n (max distance is 100 for each voter).
        worst_case = 100.0 * n
        self._round_max_payoff = worst_case  # denominator for welfare

        total_payoff = 0.0
        for agent in agents:
            payoff = -abs(winner_position - agent.ideal_point)
            total_payoff += payoff

            result = VotingRoundResult(
                my_vote=raw_votes[agent.unique_id],
                winner_id=winner_id,
                winner_position=winner_position,
                vote_counts=dict(counts),
                payoff=payoff,
                round_number=self.steps,
            )
            agent.record_result(result)

        # Social welfare: 1 + total_payoff/worst_case maps [-worst, 0] to [0, 1]
        self._round_total_payoff = worst_case + total_payoff

        self.winner_history.append(winner_id)
        self.winner_position_history.append(winner_position)
        self.vote_count_history.append(dict(counts))
        self.datacollector.collect(self)

        if self.steps >= self.n_rounds:
            self.running = False
