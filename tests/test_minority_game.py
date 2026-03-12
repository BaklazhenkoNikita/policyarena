"""Tests for Minority Game model."""

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.minority_game.brains import (
    AlwaysA,
    AlwaysB,
    RandomChoice,
    Reinforced,
    StickOrSwitch,
)
from policy_arena.games.minority_game.model import MinorityGameModel


def run_mg(brains, n_rounds=10, win_payoff=1.0, lose_payoff=-1.0, seed=42):
    """Helper: run Minority Game and return (results, model)."""
    scenario = Scenario(
        world_class=MinorityGameModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "win_payoff": win_payoff,
            "lose_payoff": lose_payoff,
        },
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    model = results.extra["model"]
    return results, model


class TestAllAlwaysA:
    """All choose A → A is majority, B (nobody) is minority.
    Everyone loses since they're all on the majority side.
    """

    def setup_method(self):
        self.results, self.model = run_mg(
            [AlwaysA(), AlwaysA(), AlwaysA()], n_rounds=10
        )

    def test_all_lose(self):
        """All agents on majority side → all lose."""
        for agent in self.model.agents:
            assert agent.cumulative_payoff == -10.0  # -1 × 10

    def test_a_fraction_one(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["a_fraction"] == 1.0

    def test_cooperation_rate_zero(self):
        """All same side → distance from 50/50 is max → cooperation_rate = 0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cooperation_rate"] == 0.0


class TestAlwaysAVsAlwaysB:
    """One A, one B → tie. Winner determined by random tiebreak."""

    def setup_method(self):
        self.results, self.model = run_mg([AlwaysA(), AlwaysB()], n_rounds=10)

    def test_a_fraction_half(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["a_fraction"] - 0.5) < 1e-9

    def test_cooperation_rate_one(self):
        """Perfect 50/50 split → cooperation_rate = 1.0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["cooperation_rate"] - 1.0) < 1e-9

    def test_total_payoff_zero(self):
        """In a tie, one wins (+1) and one loses (-1) → net 0."""
        total = sum(a.cumulative_payoff for a in self.model.agents)
        assert total == 0.0


class TestTwoAOneB:
    """2 choose A, 1 chooses B → B is minority, B wins every round.

    B agents: +1 per round = +10
    A agents: -1 per round = -10 each
    """

    def setup_method(self):
        self.results, self.model = run_mg(
            [AlwaysA(), AlwaysA(), AlwaysB()], n_rounds=10
        )

    def test_b_wins(self):
        b_agent = [a for a in self.model.agents if a.brain_name == "always_b"][0]
        assert b_agent.cumulative_payoff == 10.0

    def test_a_loses(self):
        a_agents = [a for a in self.model.agents if a.brain_name == "always_a"]
        for a in a_agents:
            assert a.cumulative_payoff == -10.0

    def test_minority_size(self):
        """1 out of 3 is minority → 1/3."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["minority_size"] - 1.0 / 3.0) < 1e-9


class TestFiveAgentsMixed:
    """3 AlwaysA + 2 AlwaysB → B is always minority."""

    def setup_method(self):
        self.results, self.model = run_mg(
            [AlwaysA(), AlwaysA(), AlwaysA(), AlwaysB(), AlwaysB()], n_rounds=10
        )

    def test_b_agents_win(self):
        b_agents = [a for a in self.model.agents if a.brain_name == "always_b"]
        for b in b_agents:
            assert b.cumulative_payoff == 10.0

    def test_a_agents_lose(self):
        a_agents = [a for a in self.model.agents if a.brain_name == "always_a"]
        for a in a_agents:
            assert a.cumulative_payoff == -10.0

    def test_minority_size(self):
        """2 out of 5 → 2/5 = 0.4."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["minority_size"] - 0.4) < 1e-9


class TestStickOrSwitchUpdates:
    """StickOrSwitch brain correctly switches after losing."""

    def test_switches_after_loss(self):
        from policy_arena.games.minority_game.types import MGRoundResult

        brain = StickOrSwitch(seed=0)
        # Force initial choice
        from policy_arena.games.minority_game.types import MGObservation

        obs = MGObservation(n_agents=3)
        first = brain.decide(obs)

        # Simulate a loss
        brain.update(
            MGRoundResult(
                choice=first,
                a_count=2,
                b_count=1,
                won=False,
                payoff=-1.0,
                round_number=0,
            )
        )
        assert brain._last_choice == (not first)

    def test_sticks_after_win(self):
        from policy_arena.games.minority_game.types import MGObservation, MGRoundResult

        brain = StickOrSwitch(seed=0)
        obs = MGObservation(n_agents=3)
        first = brain.decide(obs)

        brain.update(
            MGRoundResult(
                choice=first, a_count=1, b_count=2, won=True, payoff=1.0, round_number=0
            )
        )
        assert brain._last_choice == first


class TestReinforcedBrain:
    """Reinforced brain adjusts probability correctly."""

    def test_probability_increases_on_win_with_a(self):
        from policy_arena.games.minority_game.types import MGRoundResult

        brain = Reinforced(delta=0.1, seed=0)
        assert brain._p_a == 0.5
        brain.update(
            MGRoundResult(
                choice=True, a_count=1, b_count=2, won=True, payoff=1.0, round_number=0
            )
        )
        assert abs(brain._p_a - 0.6) < 1e-9

    def test_probability_decreases_on_loss_with_a(self):
        from policy_arena.games.minority_game.types import MGRoundResult

        brain = Reinforced(delta=0.1, seed=0)
        brain.update(
            MGRoundResult(
                choice=True,
                a_count=2,
                b_count=1,
                won=False,
                payoff=-1.0,
                round_number=0,
            )
        )
        assert abs(brain._p_a - 0.4) < 1e-9


class TestMetricsPresent:
    """Ensure all expected metrics columns exist."""

    def setup_method(self):
        self.results, self.model = run_mg(
            [AlwaysA(), AlwaysB(), RandomChoice(seed=0)], n_rounds=10
        )

    def test_all_metrics(self):
        df = self.results.model_metrics
        assert len(df) == 10
        for col in [
            "a_fraction",
            "minority_size",
            "cooperation_rate",
            "total_payoff",
            "strategy_entropy",
        ]:
            assert col in df.columns


class TestReproducibility:
    def test_deterministic(self):
        brains1 = [AlwaysA(), AlwaysB(), RandomChoice(seed=7)]
        brains2 = [AlwaysA(), AlwaysB(), RandomChoice(seed=7)]
        _, model1 = run_mg(brains1, n_rounds=20, seed=99)
        _, model2 = run_mg(brains2, n_rounds=20, seed=99)
        payoffs1 = [
            a.cumulative_payoff
            for a in sorted(model1.agents, key=lambda a: a.brain_name)
        ]
        payoffs2 = [
            a.cumulative_payoff
            for a in sorted(model2.agents, key=lambda a: a.brain_name)
        ]
        assert payoffs1 == payoffs2
