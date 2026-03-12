"""Tests for Hawk-Dove model — known agent matchups produce known payoffs and metrics."""

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.hawk_dove.brains import (
    AlwaysDove,
    AlwaysHawk,
    Bully,
    Retaliator,
)
from policy_arena.games.hawk_dove.model import (
    HawkDoveModel,
)


def run_hd(brains, n_rounds=10, seed=42):
    """Helper: run Hawk-Dove and return (results, model)."""
    scenario = Scenario(
        world_class=HawkDoveModel,
        world_params={"brains": brains, "n_rounds": n_rounds},
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    model = results.extra["model"]
    return results, model


class TestDoveVsDove:
    """Both dove → share resource (2,2) per round."""

    def setup_method(self):
        self.results, self.model = run_hd([AlwaysDove(), AlwaysDove()], n_rounds=10)

    def test_payoffs(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == 20.0  # 2 per round × 10

    def test_social_welfare_one(self):
        """(2+2)/(2+2) = 1.0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["social_welfare"] - 1.0) < 1e-9

    def test_entropy_zero(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["strategy_entropy"] == 0.0

    def test_cooperation_rate_one(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cooperation_rate"] == 1.0


class TestHawkVsHawk:
    """Both hawk → mutual injury (-1,-1) per round."""

    def setup_method(self):
        self.results, self.model = run_hd([AlwaysHawk(), AlwaysHawk()], n_rounds=10)

    def test_payoffs(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == -10.0  # -1 per round × 10

    def test_social_welfare(self):
        """(-1+-1)/(2+2) = -2/4 = -0.5."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["social_welfare"] - (-0.5)) < 1e-9

    def test_cooperation_rate_zero(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cooperation_rate"] == 0.0


class TestHawkVsDove:
    """Hawk exploits dove: hawk gets 4, dove gets 0 per round."""

    def setup_method(self):
        self.results, self.model = run_hd([AlwaysHawk(), AlwaysDove()], n_rounds=10)

    def test_hawk_payoff(self):
        hawk = [a for a in self.model.agents if a.brain_name == "always_hawk"][0]
        assert hawk.cumulative_payoff == 40.0  # 4 × 10

    def test_dove_payoff(self):
        dove = [a for a in self.model.agents if a.brain_name == "always_dove"][0]
        assert dove.cumulative_payoff == 0.0

    def test_social_welfare(self):
        """(4+0)/(2+2) = 1.0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["social_welfare"] - 1.0) < 1e-9

    def test_entropy_one(self):
        """One dove, one hawk → p=0.5 → H=1.0 bit."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["strategy_entropy"] - 1.0) < 1e-9


class TestRetaliatorVsAlwaysDove:
    """Retaliator mirrors dove → both dove every round."""

    def setup_method(self):
        self.results, self.model = run_hd([Retaliator(), AlwaysDove()], n_rounds=10)

    def test_mutual_dove(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == 20.0


class TestRetaliatorVsAlwaysHawk:
    """Retaliator cooperates R1, then retaliates with hawk.

    R1: Ret=D(dove), AH=H(hawk) → Ret gets 0, AH gets 4
    R2+: Ret=H, AH=H → both get -1
    Ret total: 0 + 9*(-1) = -9
    AH total: 4 + 9*(-1) = -5
    """

    def setup_method(self):
        self.results, self.model = run_hd([Retaliator(), AlwaysHawk()], n_rounds=10)

    def test_retaliator_payoff(self):
        ret = [a for a in self.model.agents if a.brain_name == "retaliator"][0]
        assert ret.cumulative_payoff == -9.0

    def test_hawk_payoff(self):
        hawk = [a for a in self.model.agents if a.brain_name == "always_hawk"][0]
        assert hawk.cumulative_payoff == -5.0


class TestBullyVsAlwaysDove:
    """Bully starts hawk, dove stays dove.

    R1: Bully=H, Dove=D → Bully gets 4, Dove gets 0
    R2: Bully sees dove played D(cooperate) → Bully plays H(defect)
    All rounds: Bully=H, Dove=D → Bully gets 4, Dove gets 0
    """

    def setup_method(self):
        self.results, self.model = run_hd([Bully(), AlwaysDove()], n_rounds=10)

    def test_bully_payoff(self):
        bully = [a for a in self.model.agents if a.brain_name == "bully"][0]
        assert bully.cumulative_payoff == 40.0

    def test_dove_payoff(self):
        dove = [a for a in self.model.agents if a.brain_name == "always_dove"][0]
        assert dove.cumulative_payoff == 0.0


class TestBullyVsAlwaysHawk:
    """Bully starts hawk, backs down from hawk.

    R1: Bully=H, AH=H → both get -1
    R2: Bully sees hawk played H → Bully plays D(dove); AH=H → Bully gets 0, AH gets 4
    R3: Bully sees hawk played H → Bully plays D; AH=H → same
    ...Bully=D from R2+, AH=H always
    Bully: -1 + 9*0 = -1
    AH: -1 + 9*4 = 35
    """

    def setup_method(self):
        self.results, self.model = run_hd([Bully(), AlwaysHawk()], n_rounds=10)

    def test_bully_payoff(self):
        bully = [a for a in self.model.agents if a.brain_name == "bully"][0]
        assert bully.cumulative_payoff == -1.0

    def test_hawk_payoff(self):
        hawk = [a for a in self.model.agents if a.brain_name == "always_hawk"][0]
        assert hawk.cumulative_payoff == 35.0


class TestThreeAgents:
    """Three agents produce correct metric columns and round count."""

    def setup_method(self):
        self.results, self.model = run_hd(
            [AlwaysDove(), AlwaysHawk(), Retaliator()], n_rounds=10
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
        brains1 = [Retaliator(), AlwaysHawk(), AlwaysDove()]
        brains2 = [Retaliator(), AlwaysHawk(), AlwaysDove()]
        _, model1 = run_hd(brains1, n_rounds=20, seed=99)
        _, model2 = run_hd(brains2, n_rounds=20, seed=99)
        payoffs1 = [
            a.cumulative_payoff
            for a in sorted(model1.agents, key=lambda a: a.brain_name)
        ]
        payoffs2 = [
            a.cumulative_payoff
            for a in sorted(model2.agents, key=lambda a: a.brain_name)
        ]
        assert payoffs1 == payoffs2
