"""Tests for Chicken model — known agent matchups produce known payoffs and metrics."""

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.chicken.brains import (
    AlwaysStraight,
    AlwaysSwerve,
    Brinksman,
    Cautious,
)
from policy_arena.games.chicken.model import (
    ChickenModel,
)


def run_chicken(brains, n_rounds=10, seed=42):
    """Helper: run Chicken and return (results, model)."""
    scenario = Scenario(
        world_class=ChickenModel,
        world_params={"brains": brains, "n_rounds": n_rounds},
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    model = results.extra["model"]
    return results, model


class TestSwerveVsSwerve:
    """Both swerve → (3,3) per round."""

    def setup_method(self):
        self.results, self.model = run_chicken(
            [AlwaysSwerve(), AlwaysSwerve()], n_rounds=10
        )

    def test_payoffs(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == 30.0  # 3 × 10

    def test_social_welfare_one(self):
        """(3+3)/(3+3)=1.0 — both swerve is max welfare."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["social_welfare"] - 1.0) < 1e-9

    def test_cooperation_rate_one(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cooperation_rate"] == 1.0

    def test_entropy_zero(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["strategy_entropy"] == 0.0


class TestStraightVsStraight:
    """Both straight → crash (-5,-5) per round."""

    def setup_method(self):
        self.results, self.model = run_chicken(
            [AlwaysStraight(), AlwaysStraight()], n_rounds=10
        )

    def test_payoffs(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == -50.0  # -5 × 10

    def test_social_welfare(self):
        """(-5+-5)/(3+3) = -10/6."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["social_welfare"] - (-10.0 / 6.0)) < 1e-9

    def test_cooperation_rate_zero(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cooperation_rate"] == 0.0


class TestStraightVsSwerve:
    """Straight exploits swerve: straight gets 5, swerve gets 1."""

    def setup_method(self):
        self.results, self.model = run_chicken(
            [AlwaysStraight(), AlwaysSwerve()], n_rounds=10
        )

    def test_straight_payoff(self):
        straight = [a for a in self.model.agents if a.brain_name == "always_straight"][
            0
        ]
        assert straight.cumulative_payoff == 50.0  # 5 × 10

    def test_swerve_payoff(self):
        swerve = [a for a in self.model.agents if a.brain_name == "always_swerve"][0]
        assert swerve.cumulative_payoff == 10.0  # 1 × 10

    def test_social_welfare(self):
        """(5+1)/6 = 1.0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["social_welfare"] - 1.0) < 1e-9

    def test_entropy_one(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["strategy_entropy"] - 1.0) < 1e-9


class TestCautiousVsAlwaysStraight:
    """Cautious (TFT): cooperates R1, then mirrors.

    R1: C=Swerve, AS=Straight → C gets 1, AS gets 5
    R2+: C=Straight, AS=Straight → both get -5
    C total: 1 + 9*(-5) = -44
    AS total: 5 + 9*(-5) = -40
    """

    def setup_method(self):
        self.results, self.model = run_chicken(
            [Cautious(), AlwaysStraight()], n_rounds=10
        )

    def test_cautious_payoff(self):
        c = [a for a in self.model.agents if a.brain_name == "cautious"][0]
        assert c.cumulative_payoff == -44.0

    def test_straight_payoff(self):
        s = [a for a in self.model.agents if a.brain_name == "always_straight"][0]
        assert s.cumulative_payoff == -40.0


class TestBrinksmanVsAlwaysSwerve:
    """Brinksman starts straight, swerve always swerves.

    R1: B=Straight, S=Swerve → B gets 5, S gets 1
    R2: B sees no crash (B=straight, S=swerve) → B=Straight, S=Swerve → same
    All rounds: B=Straight, S=Swerve
    B total: 5*10 = 50
    S total: 1*10 = 10
    """

    def setup_method(self):
        self.results, self.model = run_chicken(
            [Brinksman(), AlwaysSwerve()], n_rounds=10
        )

    def test_brinksman_payoff(self):
        b = [a for a in self.model.agents if a.brain_name == "brinksman"][0]
        assert b.cumulative_payoff == 50.0


class TestBrinksmanVsAlwaysStraight:
    """Brinksman vs AlwaysStraight oscillates.

    R1: B=Straight, AS=Straight → crash (-5,-5)
    R2: B backs off (crash last round) → B=Swerve, AS=Straight → B gets 1, AS gets 5
    R3: B=Straight (no crash R2, since B was swerve), AS=Straight → crash
    ...alternates
    Odd rounds (1,3,5,7,9): crash → -5,-5
    Even rounds (2,4,6,8,10): B=Swerve, AS=Straight → 1,5
    B: 5*(-5) + 5*1 = -20
    AS: 5*(-5) + 5*5 = 0
    """

    def setup_method(self):
        self.results, self.model = run_chicken(
            [Brinksman(), AlwaysStraight()], n_rounds=10
        )

    def test_brinksman_payoff(self):
        b = [a for a in self.model.agents if a.brain_name == "brinksman"][0]
        assert b.cumulative_payoff == -20.0

    def test_straight_payoff(self):
        s = [a for a in self.model.agents if a.brain_name == "always_straight"][0]
        assert s.cumulative_payoff == 0.0


class TestThreeAgents:
    """Three agents produce correct metrics."""

    def setup_method(self):
        self.results, self.model = run_chicken(
            [AlwaysSwerve(), AlwaysStraight(), Cautious()], n_rounds=10
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


class TestReproducibility:
    def test_deterministic(self):
        brains1 = [Cautious(), AlwaysStraight(), AlwaysSwerve()]
        brains2 = [Cautious(), AlwaysStraight(), AlwaysSwerve()]
        _, model1 = run_chicken(brains1, n_rounds=20, seed=99)
        _, model2 = run_chicken(brains2, n_rounds=20, seed=99)
        payoffs1 = [
            a.cumulative_payoff
            for a in sorted(model1.agents, key=lambda a: a.brain_name)
        ]
        payoffs2 = [
            a.cumulative_payoff
            for a in sorted(model2.agents, key=lambda a: a.brain_name)
        ]
        assert payoffs1 == payoffs2
