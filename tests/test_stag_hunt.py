"""Tests for Stag Hunt model — known agent matchups produce known payoffs and metrics."""

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.stag_hunt.brains import (
    AlwaysHare,
    AlwaysStag,
    CautiousStag,
    TrustButVerify,
)
from policy_arena.games.stag_hunt.model import (
    StagHuntModel,
)


def run_sh(brains, n_rounds=10, seed=42):
    """Helper: run Stag Hunt and return (results, model)."""
    scenario = Scenario(
        world_class=StagHuntModel,
        world_params={"brains": brains, "n_rounds": n_rounds},
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    model = results.extra["model"]
    return results, model


class TestAllStagVsAllStag:
    """SS every round → both NE (all-stag), welfare=1.0, entropy=0."""

    def setup_method(self):
        self.results, self.model = run_sh([AlwaysStag(), AlwaysStag()], n_rounds=10)

    def test_payoffs(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == 40.0  # 4 per round × 10

    def test_nash_distance_zero(self):
        """All-stag is a NE → distance = 0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["nash_eq_distance"] == 0.0

    def test_social_welfare_one(self):
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


class TestAllHareVsAllHare:
    """HH every round → also NE, welfare = 4/8 = 0.5, entropy=0."""

    def setup_method(self):
        self.results, self.model = run_sh([AlwaysHare(), AlwaysHare()], n_rounds=10)

    def test_payoffs(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == 20.0  # 2 per round × 10

    def test_nash_distance_zero(self):
        """All-hare is also a NE → distance = 0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["nash_eq_distance"] == 0.0

    def test_social_welfare(self):
        """2+2=4 per pair, max=4+4=8 → welfare=0.5."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["social_welfare"] - 0.5) < 1e-9

    def test_cooperation_rate_zero(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cooperation_rate"] == 0.0


class TestStagVsHare:
    """Exploitation: stag hunter gets 0, hare hunter gets 3 per round."""

    def setup_method(self):
        self.results, self.model = run_sh([AlwaysStag(), AlwaysHare()], n_rounds=10)

    def test_stag_payoff(self):
        stag = [a for a in self.model.agents if a.brain_name == "always_stag"][0]
        assert stag.cumulative_payoff == 0.0

    def test_hare_payoff(self):
        hare = [a for a in self.model.agents if a.brain_name == "always_hare"][0]
        assert hare.cumulative_payoff == 30.0  # 3 per round × 10

    def test_cooperation_rate(self):
        """1 stag, 1 hare → 0.5."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["cooperation_rate"] - 0.5) < 1e-9

    def test_social_welfare(self):
        """0+3=3 per pair, max=8 → 3/8."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["social_welfare"] - 3.0 / 8.0) < 1e-9


class TestTrustButVerifyVsAlwaysHare:
    """TBV cooperates R1, then mirrors → hare forever after R1.

    R1: TBV=S, AH=H → TBV gets 0, AH gets 3
    R2+: TBV=H, AH=H → both get 2
    TBV total: 0 + 9*2 = 18
    AH total: 3 + 9*2 = 21
    """

    def setup_method(self):
        self.results, self.model = run_sh([TrustButVerify(), AlwaysHare()], n_rounds=10)

    def test_tbv_payoff(self):
        tbv = [a for a in self.model.agents if a.brain_name == "trust_but_verify"][0]
        assert tbv.cumulative_payoff == 18.0

    def test_hare_payoff(self):
        ah = [a for a in self.model.agents if a.brain_name == "always_hare"][0]
        assert ah.cumulative_payoff == 21.0

    def test_first_round_cooperation_rate(self):
        """R1: TBV=stag, AH=hare → 0.5."""
        df = self.results.model_metrics
        assert abs(df.iloc[0]["cooperation_rate"] - 0.5) < 1e-9

    def test_later_rounds_cooperation_rate(self):
        """R2+: both hare → 0.0."""
        df = self.results.model_metrics
        for i in range(1, len(df)):
            assert df.iloc[i]["cooperation_rate"] == 0.0


class TestTrustButVerifyVsAlwaysStag:
    """TBV mirrors stag → both cooperate every round."""

    def setup_method(self):
        self.results, self.model = run_sh([TrustButVerify(), AlwaysStag()], n_rounds=10)

    def test_mutual_cooperation(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == 40.0


class TestCautiousStagVsAlwaysStag:
    """Cautious starts hare, then switches to stag.

    R1: CS=H, AS=S → CS gets 3, AS gets 0
    R2+: CS=S, AS=S → both get 4
    CS total: 3 + 9*4 = 39
    AS total: 0 + 9*4 = 36
    """

    def setup_method(self):
        self.results, self.model = run_sh([CautiousStag(), AlwaysStag()], n_rounds=10)

    def test_cautious_payoff(self):
        cs = [a for a in self.model.agents if a.brain_name == "cautious_stag"][0]
        assert cs.cumulative_payoff == 39.0

    def test_stag_payoff(self):
        s = [a for a in self.model.agents if a.brain_name == "always_stag"][0]
        assert s.cumulative_payoff == 36.0


class TestThreeAgentsTournament:
    """Three agents: AlwaysStag, AlwaysHare, TrustButVerify."""

    def setup_method(self):
        self.results, self.model = run_sh(
            [AlwaysStag(), AlwaysHare(), TrustButVerify()], n_rounds=10
        )

    def test_all_metrics_present(self):
        df = self.results.model_metrics
        assert len(df) == 10
        for col in [
            "cooperation_rate",
            "nash_eq_distance",
            "social_welfare",
            "strategy_entropy",
        ]:
            assert col in df.columns

    def test_agent_count(self):
        assert len(list(self.model.agents)) == 3


class TestReproducibility:
    """Same seed → same results."""

    def test_deterministic(self):
        brains1 = [TrustButVerify(), AlwaysHare(), AlwaysStag()]
        brains2 = [TrustButVerify(), AlwaysHare(), AlwaysStag()]
        _, model1 = run_sh(brains1, n_rounds=20, seed=99)
        _, model2 = run_sh(brains2, n_rounds=20, seed=99)
        payoffs1 = [
            a.cumulative_payoff
            for a in sorted(model1.agents, key=lambda a: a.brain_name)
        ]
        payoffs2 = [
            a.cumulative_payoff
            for a in sorted(model2.agents, key=lambda a: a.brain_name)
        ]
        assert payoffs1 == payoffs2
