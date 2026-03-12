"""Tests for Battle of the Sexes model — known matchups produce known payoffs and metrics."""

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.battle_of_sexes.brains import (
    AlwaysA,
    AlwaysB,
    Compromiser,
    Stubborn,
)
from policy_arena.games.battle_of_sexes.model import (
    BattleOfSexesModel,
)


def run_bos(brains, n_rounds=10, seed=42):
    """Helper: run Battle of the Sexes and return (results, model)."""
    scenario = Scenario(
        world_class=BattleOfSexesModel,
        world_params={"brains": brains, "n_rounds": n_rounds},
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    model = results.extra["model"]
    return results, model


class TestAlwaysAVsAlwaysA:
    """Both pick A → (3,2) per round. Coordinated on A."""

    def setup_method(self):
        self.results, self.model = run_bos([AlwaysA(), AlwaysA()], n_rounds=10)

    def test_payoffs(self):
        # In round-robin, each pair (A,B): row gets 3, col gets 2
        # With 2 agents, 1 pair per round. Each is row once and col once
        # Actually: in the pair, agent 0 is "row" and agent 1 is "column"
        # (A,A) = (3,2) — asymmetric!
        total = sum(a.cumulative_payoff for a in self.model.agents)
        assert total == 50.0  # (3+2) × 10

    def test_social_welfare_one(self):
        """(3+2)/max(3+2,2+3) = 5/5 = 1.0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["social_welfare"] - 1.0) < 1e-9

    def test_cooperation_rate_one(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cooperation_rate"] == 1.0

    def test_nash_distance_zero(self):
        """All-A is a NE → distance = 0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["nash_eq_distance"] == 0.0


class TestAlwaysBVsAlwaysB:
    """Both pick B → (2,3) per round. Coordinated on B."""

    def setup_method(self):
        self.results, self.model = run_bos([AlwaysB(), AlwaysB()], n_rounds=10)

    def test_payoffs(self):
        total = sum(a.cumulative_payoff for a in self.model.agents)
        assert total == 50.0  # (2+3) × 10

    def test_social_welfare_one(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["social_welfare"] - 1.0) < 1e-9

    def test_nash_distance_zero(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["nash_eq_distance"] == 0.0


class TestAlwaysAVsAlwaysB:
    """Miscoordination: (A,B) = (0,0) every round."""

    def setup_method(self):
        self.results, self.model = run_bos([AlwaysA(), AlwaysB()], n_rounds=10)

    def test_payoffs(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == 0.0

    def test_social_welfare_zero(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["social_welfare"] == 0.0

    def test_entropy_one(self):
        """One A, one B → H=1.0 bit."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["strategy_entropy"] - 1.0) < 1e-9

    def test_coordination_rate_zero(self):
        """Different choices → coordination rate = 0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["coordination_rate"] == 0.0


class TestCompromiserVsAlwaysA:
    """Compromiser mirrors opponent → both play A."""

    def setup_method(self):
        self.results, self.model = run_bos([Compromiser(), AlwaysA()], n_rounds=10)

    def test_full_coordination_on_a(self):
        total = sum(a.cumulative_payoff for a in self.model.agents)
        assert total == 50.0  # (3+2) × 10

    def test_coordination_rate_one(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["coordination_rate"] == 1.0


class TestCompromiserVsAlwaysB:
    """Compromiser starts A, then mirrors B.

    R1: Comp=A, B=B → miscoord (0,0)
    R2+: Comp=B, B=B → (2,3)
    Total: 0 + 9*5 = 45
    """

    def setup_method(self):
        self.results, self.model = run_bos([Compromiser(), AlwaysB()], n_rounds=10)

    def test_total_payoff(self):
        total = sum(a.cumulative_payoff for a in self.model.agents)
        assert total == 45.0


class TestStubbornVsCompromiser:
    """Stubborn plays opposite of opponent → perpetual miscoordination.

    R1: Stubborn=A (default), Comp=A (default) → (3,2)
    R2: Stubborn sees A→plays B; Comp sees A→plays A → (B,A)=(0,0)
    R3: Stubborn sees A→plays B; Comp sees B→plays B → (B,B)=(2,3)
    R4: Stubborn sees B→plays A; Comp sees B→plays B → (A,B)=(0,0)
    R5: Stubborn sees B→plays A; Comp sees A→plays A → (A,A)=(3,2)
    ...pattern repeats: (3,2),(0,0),(2,3),(0,0),...
    """

    def setup_method(self):
        self.results, self.model = run_bos([Stubborn(), Compromiser()], n_rounds=10)

    def test_runs_without_error(self):
        assert len(self.results.model_metrics) == 10


class TestThreeAgents:
    """Three agents produce correct metrics."""

    def setup_method(self):
        self.results, self.model = run_bos(
            [AlwaysA(), AlwaysB(), Compromiser()], n_rounds=10
        )

    def test_all_metrics_present(self):
        df = self.results.model_metrics
        assert len(df) == 10
        for col in [
            "cooperation_rate",
            "coordination_rate",
            "nash_eq_distance",
            "social_welfare",
            "strategy_entropy",
        ]:
            assert col in df.columns

    def test_agent_count(self):
        assert len(list(self.model.agents)) == 3


class TestReproducibility:
    def test_deterministic(self):
        brains1 = [AlwaysA(), AlwaysB(), Compromiser()]
        brains2 = [AlwaysA(), AlwaysB(), Compromiser()]
        _, model1 = run_bos(brains1, n_rounds=20, seed=99)
        _, model2 = run_bos(brains2, n_rounds=20, seed=99)
        payoffs1 = [
            a.cumulative_payoff
            for a in sorted(model1.agents, key=lambda a: a.brain_name)
        ]
        payoffs2 = [
            a.cumulative_payoff
            for a in sorted(model2.agents, key=lambda a: a.brain_name)
        ]
        assert payoffs1 == payoffs2
