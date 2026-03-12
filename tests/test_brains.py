"""Tests for rule-based brains — deterministic strategies produce known outputs."""

from policy_arena.brains.rule_based import (
    AlwaysCooperate,
    AlwaysDefect,
    Pavlov,
    RandomBrain,
    TitForTat,
)
from policy_arena.core.types import Action, Observation


class TestAlwaysCooperate:
    def test_always_cooperates(self):
        brain = AlwaysCooperate()
        for _ in range(10):
            assert brain.decide(Observation()) == Action.COOPERATE

    def test_ignores_history(self):
        brain = AlwaysCooperate()
        obs = Observation(
            opponent_history=[Action.DEFECT] * 5,
            my_history=[Action.COOPERATE] * 5,
        )
        assert brain.decide(obs) == Action.COOPERATE


class TestAlwaysDefect:
    def test_always_defects(self):
        brain = AlwaysDefect()
        for _ in range(10):
            assert brain.decide(Observation()) == Action.DEFECT

    def test_ignores_history(self):
        brain = AlwaysDefect()
        obs = Observation(
            opponent_history=[Action.COOPERATE] * 5,
            my_history=[Action.DEFECT] * 5,
        )
        assert brain.decide(obs) == Action.DEFECT


class TestTitForTat:
    def test_cooperates_first(self):
        brain = TitForTat()
        assert brain.decide(Observation()) == Action.COOPERATE

    def test_copies_opponent_cooperate(self):
        brain = TitForTat()
        obs = Observation(
            opponent_history=[Action.COOPERATE],
            my_history=[Action.COOPERATE],
        )
        assert brain.decide(obs) == Action.COOPERATE

    def test_copies_opponent_defect(self):
        brain = TitForTat()
        obs = Observation(
            opponent_history=[Action.COOPERATE, Action.DEFECT],
            my_history=[Action.COOPERATE, Action.COOPERATE],
        )
        assert brain.decide(obs) == Action.DEFECT

    def test_forgives_after_cooperate(self):
        brain = TitForTat()
        obs = Observation(
            opponent_history=[Action.DEFECT, Action.COOPERATE],
            my_history=[Action.COOPERATE, Action.DEFECT],
        )
        assert brain.decide(obs) == Action.COOPERATE


class TestPavlov:
    def test_cooperates_first(self):
        brain = Pavlov()
        assert brain.decide(Observation()) == Action.COOPERATE

    def test_repeats_on_mutual_cooperate(self):
        """CC -> cooperate (same action = cooperate)."""
        brain = Pavlov()
        obs = Observation(
            my_history=[Action.COOPERATE],
            opponent_history=[Action.COOPERATE],
        )
        assert brain.decide(obs) == Action.COOPERATE

    def test_switches_on_exploited(self):
        """CD -> defect (different actions = defect)."""
        brain = Pavlov()
        obs = Observation(
            my_history=[Action.COOPERATE],
            opponent_history=[Action.DEFECT],
        )
        assert brain.decide(obs) == Action.DEFECT

    def test_switches_on_mutual_defect(self):
        """DD -> cooperate (same action = cooperate)."""
        brain = Pavlov()
        obs = Observation(
            my_history=[Action.DEFECT],
            opponent_history=[Action.DEFECT],
        )
        assert brain.decide(obs) == Action.COOPERATE

    def test_repeats_on_exploit_success(self):
        """DC -> defect (different actions = defect)."""
        brain = Pavlov()
        obs = Observation(
            my_history=[Action.DEFECT],
            opponent_history=[Action.COOPERATE],
        )
        assert brain.decide(obs) == Action.DEFECT


class TestRandomBrain:
    def test_deterministic_with_seed(self):
        brain1 = RandomBrain(cooperation_probability=0.5, seed=42)
        brain2 = RandomBrain(cooperation_probability=0.5, seed=42)
        obs = Observation()
        for _ in range(20):
            assert brain1.decide(obs) == brain2.decide(obs)

    def test_always_cooperate_at_p1(self):
        brain = RandomBrain(cooperation_probability=1.0, seed=0)
        for _ in range(50):
            assert brain.decide(Observation()) == Action.COOPERATE

    def test_always_defect_at_p0(self):
        brain = RandomBrain(cooperation_probability=0.0, seed=0)
        for _ in range(50):
            assert brain.decide(Observation()) == Action.DEFECT
