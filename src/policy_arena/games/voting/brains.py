"""Rule-based brains for the Voting & Election Game."""

from __future__ import annotations

import random
from typing import Any

from policy_arena.brains.base import Brain
from policy_arena.games.voting.types import VotingObservation, VotingRoundResult


class SincereVoter(Brain):
    """Always votes for the closest candidate.

    Plurality: vote for nearest candidate.
    Approval: approve all candidates within a threshold distance.
    Borda: rank candidates by distance (closest first).
    """

    def __init__(self, approval_threshold: float = 25.0):
        self._approval_threshold = approval_threshold

    @property
    def name(self) -> str:
        return "sincere_voter"

    def decide(self, observation: VotingObservation) -> Any:
        positions = observation.candidate_positions
        ideal = observation.my_ideal_point
        distances = [
            (abs(positions[c] - ideal), c) for c in range(observation.n_candidates)
        ]
        distances.sort()

        if observation.voting_rule == "plurality":
            return distances[0][1]
        elif observation.voting_rule == "approval":
            return [c for d, c in distances if d <= self._approval_threshold]
        else:  # borda
            return [c for _, c in distances]

    def update(self, result: VotingRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class StrategicVoter(Brain):
    """Votes strategically under plurality; sincere under other rules.

    Plurality: if the closest candidate is not among the top-2 from last round,
    vote for the preferred candidate among the top-2 instead.
    Approval/Borda: votes sincerely.
    """

    def __init__(self, approval_threshold: float = 25.0):
        self._approval_threshold = approval_threshold

    @property
    def name(self) -> str:
        return "strategic_voter"

    def decide(self, observation: VotingObservation) -> Any:
        positions = observation.candidate_positions
        ideal = observation.my_ideal_point
        distances = [
            (abs(positions[c] - ideal), c) for c in range(observation.n_candidates)
        ]
        distances.sort()

        if observation.voting_rule == "plurality":
            favorite = distances[0][1]

            if observation.all_vote_counts:
                last_counts = observation.all_vote_counts[-1]
                # Find top-2 candidates by vote count
                sorted_candidates = sorted(
                    last_counts.keys(),
                    key=lambda c: last_counts.get(c, 0),
                    reverse=True,
                )
                top_2 = set(sorted_candidates[:2])

                if favorite not in top_2:
                    # Vote for the closest candidate among the top-2
                    top_2_by_distance = sorted(
                        top_2, key=lambda c: abs(positions[c] - ideal)
                    )
                    return top_2_by_distance[0]

            return favorite

        elif observation.voting_rule == "approval":
            return [c for d, c in distances if d <= self._approval_threshold]
        else:  # borda
            return [c for _, c in distances]

    def update(self, result: VotingRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class RandomVoter(Brain):
    """Casts a random valid vote."""

    @property
    def name(self) -> str:
        return "random_voter"

    def decide(self, observation: VotingObservation) -> Any:
        n = observation.n_candidates
        if observation.voting_rule == "plurality":
            return random.randint(0, n - 1)
        elif observation.voting_rule == "approval":
            # Approve a random non-empty subset
            subset = [c for c in range(n) if random.random() > 0.5]
            if not subset:
                subset = [random.randint(0, n - 1)]
            return subset
        else:  # borda
            ranking = list(range(n))
            random.shuffle(ranking)
            return ranking

    def update(self, result: VotingRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class ContrarianVoter(Brain):
    """Votes for the least popular candidate from the last round.

    If no history, votes randomly.
    """

    @property
    def name(self) -> str:
        return "contrarian_voter"

    def decide(self, observation: VotingObservation) -> Any:
        n = observation.n_candidates

        if not observation.all_vote_counts:
            # No history: vote randomly
            if observation.voting_rule == "plurality":
                return random.randint(0, n - 1)
            elif observation.voting_rule == "approval":
                return [random.randint(0, n - 1)]
            else:
                ranking = list(range(n))
                random.shuffle(ranking)
                return ranking

        last_counts = observation.all_vote_counts[-1]
        # Sort by ascending vote count (least popular first)
        sorted_candidates = sorted(range(n), key=lambda c: last_counts.get(c, 0))

        if observation.voting_rule == "plurality":
            return sorted_candidates[0]
        elif observation.voting_rule == "approval":
            # Approve the bottom half
            half = max(1, n // 2)
            return sorted_candidates[:half]
        else:  # borda
            return sorted_candidates  # rank least popular first (contrarian ranking)

    def update(self, result: VotingRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass
