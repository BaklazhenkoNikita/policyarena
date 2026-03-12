"""Tests for PD model — known agent matchups produce known payoffs and metrics."""

from policy_arena.brains.rule_based import (
    AlwaysCooperate,
    AlwaysDefect,
    Pavlov,
    TitForTat,
)
from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.core.types import Action
from policy_arena.games.prisoners_dilemma.model import (
    DEFAULT_PAYOFF_MATRIX,
    PrisonersDilemmaModel,
)
from policy_arena.metrics.regret import compute_individual_regret


def run_pd(brains, n_rounds=10, seed=42):
    """Helper: run PD and return (results, model)."""
    scenario = Scenario(
        world_class=PrisonersDilemmaModel,
        world_params={"brains": brains, "n_rounds": n_rounds},
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    model = results.extra["model"]
    return results, model


class TestAllDefectVsAllDefect:
    """DD every round → NE distance=0, social welfare=1/6, entropy=0."""

    def setup_method(self):
        self.results, self.model = run_pd([AlwaysDefect(), AlwaysDefect()], n_rounds=10)

    def test_payoffs(self):
        agents = list(self.model.agents)
        for agent in agents:
            assert agent.cumulative_payoff == 10.0  # 1 per round × 10 rounds

    def test_nash_distance_zero(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["nash_eq_distance"] == 0.0

    def test_social_welfare(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["social_welfare"] - 1 / 3) < 1e-9

    def test_entropy_zero(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["strategy_entropy"] == 0.0

    def test_cooperation_rate_zero(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cooperation_rate"] == 0.0

    def test_zero_regret(self):
        for agent in self.model.agents:
            assert compute_individual_regret(agent, DEFAULT_PAYOFF_MATRIX) == 0.0


class TestAllCoopVsAllCoop:
    """CC every round → NE distance=1, social welfare=1.0, entropy=0."""

    def setup_method(self):
        self.results, self.model = run_pd(
            [AlwaysCooperate(), AlwaysCooperate()], n_rounds=10
        )

    def test_payoffs(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == 30.0  # 3 per round × 10 rounds

    def test_nash_distance_one(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["nash_eq_distance"] == 1.0

    def test_social_welfare_one(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["social_welfare"] - 1.0) < 1e-9

    def test_entropy_zero(self):
        """Both cooperate → single action → entropy=0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["strategy_entropy"] == 0.0

    def test_regret(self):
        """AllCoop could have done 5/round defecting against AllCoop → regret=20."""
        for agent in self.model.agents:
            regret = compute_individual_regret(agent, DEFAULT_PAYOFF_MATRIX)
            assert regret == 20.0  # 50 (defect) - 30 (cooperate) = 20


class TestAllDefectVsAllCoop:
    """Exploitation: defector gets 5/round, cooperator gets 0/round."""

    def setup_method(self):
        self.results, self.model = run_pd(
            [AlwaysDefect(), AlwaysCooperate()], n_rounds=10
        )

    def test_defector_payoff(self):
        agents = sorted(
            self.model.agents, key=lambda a: a.cumulative_payoff, reverse=True
        )
        defector = agents[0]
        assert defector.brain_name == "always_defect"
        assert defector.cumulative_payoff == 50.0

    def test_cooperator_payoff(self):
        agents = sorted(self.model.agents, key=lambda a: a.cumulative_payoff)
        cooperator = agents[0]
        assert cooperator.brain_name == "always_cooperate"
        assert cooperator.cumulative_payoff == 0.0

    def test_social_welfare(self):
        """5+0=5 per round, max=6 → welfare=5/6."""
        df = self.results.model_metrics
        expected = 5.0 / 6.0
        for _, row in df.iterrows():
            assert abs(row["social_welfare"] - expected) < 1e-9

    def test_entropy_one(self):
        """One C, one D → p=0.5 each → H=1.0 bit."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["strategy_entropy"] - 1.0) < 1e-9

    def test_cooperation_rate(self):
        """1 out of 2 actions is cooperate → 0.5."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["cooperation_rate"] - 0.5) < 1e-9


class TestTFTvsAlwaysDefect:
    """TFT cooperates round 1, then defects forever.

    Round 1: TFT=C, AD=D → TFT gets 0, AD gets 5
    Rounds 2-10: TFT=D, AD=D → both get 1
    TFT total: 0 + 9*1 = 9
    AD total: 5 + 9*1 = 14
    """

    def setup_method(self):
        self.results, self.model = run_pd([TitForTat(), AlwaysDefect()], n_rounds=10)

    def test_tft_payoff(self):
        tft = [a for a in self.model.agents if a.brain_name == "tit_for_tat"][0]
        assert tft.cumulative_payoff == 9.0

    def test_defector_payoff(self):
        ad = [a for a in self.model.agents if a.brain_name == "always_defect"][0]
        assert ad.cumulative_payoff == 14.0

    def test_first_round_cooperation_rate(self):
        """Round 1: TFT cooperates, AD defects → rate=0.5."""
        df = self.results.model_metrics
        assert abs(df.iloc[0]["cooperation_rate"] - 0.5) < 1e-9

    def test_later_rounds_cooperation_rate(self):
        """Rounds 2+: both defect → rate=0."""
        df = self.results.model_metrics
        for i in range(1, len(df)):
            assert df.iloc[i]["cooperation_rate"] == 0.0


class TestTFTvsAllCoop:
    """TFT mirrors AllCoop → both cooperate every round.

    Both get 3/round × 10 = 30.
    """

    def setup_method(self):
        self.results, self.model = run_pd([TitForTat(), AlwaysCooperate()], n_rounds=10)

    def test_mutual_cooperation(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == 30.0

    def test_cooperation_rate_always_one(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cooperation_rate"] == 1.0


class TestTFTvsTFT:
    """Two TFTs → both cooperate first round, then mirror → CC forever."""

    def setup_method(self):
        self.results, self.model = run_pd([TitForTat(), TitForTat()], n_rounds=10)

    def test_mutual_cooperation(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == 30.0


class TestPavlovDynamics:
    """Pavlov vs AlwaysDefect.

    R1: Pavlov=C, AD=D → Pavlov gets 0 (different → switch to D)
    R2: Pavlov=D, AD=D → both get 1 (same → switch to C)
    R3: Pavlov=C, AD=D → Pavlov gets 0 (different → switch to D)
    ...oscillates: C,D,C,D,...
    """

    def setup_method(self):
        self.results, self.model = run_pd([Pavlov(), AlwaysDefect()], n_rounds=10)

    def test_pavlov_oscillates(self):
        pavlov = [a for a in self.model.agents if a.brain_name == "pavlov"][0]
        ad = [a for a in self.model.agents if a.brain_name == "always_defect"][0]
        ad_id = ad.unique_id
        history = pavlov._my_history[ad_id]
        expected = [Action.COOPERATE, Action.DEFECT] * 5
        assert history == expected

    def test_pavlov_payoff(self):
        """5 rounds of C vs D (0) + 5 rounds of D vs D (1) = 5."""
        pavlov = [a for a in self.model.agents if a.brain_name == "pavlov"][0]
        assert pavlov.cumulative_payoff == 5.0


class TestEngineReproducibility:
    """Same seed → same results."""

    def test_deterministic(self):
        brains1 = [TitForTat(), AlwaysDefect(), AlwaysCooperate()]
        brains2 = [TitForTat(), AlwaysDefect(), AlwaysCooperate()]
        _, model1 = run_pd(brains1, n_rounds=50, seed=99)
        _, model2 = run_pd(brains2, n_rounds=50, seed=99)
        payoffs1 = [
            a.cumulative_payoff
            for a in sorted(model1.agents, key=lambda a: a.brain_name)
        ]
        payoffs2 = [
            a.cumulative_payoff
            for a in sorted(model2.agents, key=lambda a: a.brain_name)
        ]
        assert payoffs1 == payoffs2


class TestMetricsEdgeCases:
    def test_single_agent_no_interactions(self):
        """1 agent → no pairs → metrics should handle gracefully."""
        results, model = run_pd([AlwaysCooperate()], n_rounds=5)
        df = results.model_metrics
        assert len(df) == 5
        for _, row in df.iterrows():
            assert row["cooperation_rate"] == 0.0
            assert row["nash_eq_distance"] == 0.0
            assert row["strategy_entropy"] == 0.0
