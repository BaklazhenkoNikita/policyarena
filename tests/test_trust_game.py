"""Tests for Trust Game model — known brain matchups produce known payoffs and metrics."""

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.trust_game.brains import (
    Exploiter,
    FairPlayer,
    FullTrust,
    NoTrust,
)
from policy_arena.games.trust_game.model import TrustGameModel


def run_tg(brains, n_rounds=10, endowment=10.0, multiplier=3.0, seed=42):
    """Helper: run Trust Game and return (results, model)."""
    scenario = Scenario(
        world_class=TrustGameModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "endowment": endowment,
            "multiplier": multiplier,
        },
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    model = results.extra["model"]
    return results, model


class TestNoTrustVsNoTrust:
    """NE strategy: invest 0, return 0.

    Both invest 0 → no transfers.
    Investor payoff = endowment - 0 + 0 = 10
    Trustee payoff = 0 * multiplier - 0 = 0
    Each agent is investor once and trustee once per round (permutations).
    Per round per agent: 10 (as investor) + 0 (as trustee) = 10
    Over 10 rounds: 100
    """

    def setup_method(self):
        self.results, self.model = run_tg([NoTrust(), NoTrust()], n_rounds=10)

    def test_payoffs(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == 100.0  # 10 per round × 10

    def test_nash_distance_zero(self):
        """NE is invest 0 → distance = 0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["nash_eq_distance"] == 0.0

    def test_cooperation_rate_zero(self):
        """No investment → cooperation rate = 0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cooperation_rate"] == 0.0


class TestFullTrustVsFullTrust:
    """Both invest full endowment, return 1/3 of received.

    Pair (A invests in B):
    A invests 10, B receives 30, B returns 10.
    A payoff: 10 - 10 + 10 = 10
    B payoff: 30 - 10 = 20

    Each agent: 10 (as investor) + 20 (as trustee) = 30 per round
    Over 10 rounds: 300
    """

    def setup_method(self):
        self.results, self.model = run_tg([FullTrust(), FullTrust()], n_rounds=10)

    def test_payoffs(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == 300.0

    def test_cooperation_rate_one(self):
        """Full investment → cooperation rate = 1.0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert abs(row["cooperation_rate"] - 1.0) < 1e-9


class TestFullTrustVsExploiter:
    """FullTrust invests all and returns 1/3. Exploiter invests half, returns 0.

    Pair (FT invests in E):
    FT invests 10, E receives 30, E returns 0.
    FT payoff as investor: 10 - 10 + 0 = 0
    E payoff as trustee: 30 - 0 = 30

    Pair (E invests in FT):
    E invests 5, FT receives 15, FT returns 5.
    E payoff as investor: 10 - 5 + 5 = 10
    FT payoff as trustee: 15 - 5 = 10

    Per round: FT = 0+10 = 10, E = 30+10 = 40
    Over 10 rounds: FT = 100, E = 400
    """

    def setup_method(self):
        self.results, self.model = run_tg([FullTrust(), Exploiter()], n_rounds=10)

    def test_full_trust_payoff(self):
        ft = [a for a in self.model.agents if a.brain_name == "full_trust"][0]
        assert ft.cumulative_payoff == 100.0

    def test_exploiter_payoff(self):
        e = [a for a in self.model.agents if a.brain_name == "exploiter"][0]
        assert e.cumulative_payoff == 400.0


class TestFairPlayerVsFairPlayer:
    """Both invest 50%, return 50% of received.

    Pair (A invests in B):
    A invests 5, B receives 15, B returns 7.5.
    A payoff: 10 - 5 + 7.5 = 12.5
    B payoff: 15 - 7.5 = 7.5

    Each agent: 12.5 + 7.5 = 20 per round
    Over 10 rounds: 200
    """

    def setup_method(self):
        self.results, self.model = run_tg([FairPlayer(), FairPlayer()], n_rounds=10)

    def test_payoffs(self):
        for agent in self.model.agents:
            assert abs(agent.cumulative_payoff - 200.0) < 1e-9


class TestNoTrustVsFullTrust:
    """NoTrust invests 0 and returns 0.

    Pair (NT invests in FT): NT invests 0 → FT receives 0, returns 0.
    NT investor payoff: 10 - 0 + 0 = 10
    FT trustee payoff: 0 - 0 = 0

    Pair (FT invests in NT): FT invests 10 → NT receives 30, returns 0.
    FT investor payoff: 10 - 10 + 0 = 0
    NT trustee payoff: 30 - 0 = 30

    Per round: NT = 10+30 = 40, FT = 0+0 = 0
    Over 10 rounds: NT = 400, FT = 0
    """

    def setup_method(self):
        self.results, self.model = run_tg([NoTrust(), FullTrust()], n_rounds=10)

    def test_no_trust_payoff(self):
        nt = [a for a in self.model.agents if a.brain_name == "no_trust"][0]
        assert nt.cumulative_payoff == 400.0

    def test_full_trust_payoff(self):
        ft = [a for a in self.model.agents if a.brain_name == "full_trust"][0]
        assert ft.cumulative_payoff == 0.0


class TestMetricsPresent:
    """Ensure all expected metrics are reported."""

    def setup_method(self):
        self.results, self.model = run_tg(
            [FullTrust(), FairPlayer(), NoTrust()], n_rounds=10
        )

    def test_all_metrics(self):
        df = self.results.model_metrics
        assert len(df) == 10
        for col in [
            "cooperation_rate",
            "avg_return_rate",
            "nash_eq_distance",
            "social_welfare",
            "strategy_entropy",
        ]:
            assert col in df.columns

    def test_agent_count(self):
        assert len(list(self.model.agents)) == 3


class TestReproducibility:
    def test_deterministic(self):
        brains1 = [FullTrust(), NoTrust(), FairPlayer()]
        brains2 = [FullTrust(), NoTrust(), FairPlayer()]
        _, model1 = run_tg(brains1, n_rounds=20, seed=99)
        _, model2 = run_tg(brains2, n_rounds=20, seed=99)
        payoffs1 = [
            a.cumulative_payoff
            for a in sorted(model1.agents, key=lambda a: a.brain_name)
        ]
        payoffs2 = [
            a.cumulative_payoff
            for a in sorted(model2.agents, key=lambda a: a.brain_name)
        ]
        assert payoffs1 == payoffs2
