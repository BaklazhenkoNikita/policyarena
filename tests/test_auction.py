"""Tests for the Sealed-Bid Auction model."""

import pytest

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.auction.brains import (
    AggressiveBidder,
    BestResponseBidder,
    ShadedBidder,
    TruthfulBidder,
)
from policy_arena.games.auction.model import AuctionModel, _bin_bid


def run_auction(
    brains,
    n_rounds=10,
    auction_type="first_price",
    value_min=0.0,
    value_max=100.0,
    max_bid=150.0,
    seed=42,
):
    scenario = Scenario(
        world_class=AuctionModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "auction_type": auction_type,
            "value_min": value_min,
            "value_max": value_max,
            "max_bid": max_bid,
        },
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    model = results.extra["model"]
    return results, model


class TestBinBid:
    def test_zero_max(self):
        assert _bin_bid(10, 0) == "0%"

    def test_bins(self):
        assert _bin_bid(0, 100) == "0%"
        assert _bin_bid(20, 100) == "15%"
        assert _bin_bid(50, 100) == "45%"
        assert _bin_bid(100, 100) == "100%"


class TestFirstPriceTruthful:
    """Truthful bidders in first-price auction."""

    def setup_method(self):
        self.results, self.model = run_auction(
            [TruthfulBidder(), TruthfulBidder(), TruthfulBidder()],
            n_rounds=10,
            auction_type="first_price",
        )

    def test_runs(self):
        assert len(self.results.model_metrics) == 10

    def test_winner_pays_own_bid(self):
        """In first-price, winner pays their bid."""
        for wb, pp in zip(
            self.model.winning_bid_history, self.model.price_paid_history, strict=False
        ):
            assert wb == pytest.approx(pp)


class TestSecondPriceTruthful:
    """Truthful bidders in second-price auction."""

    def setup_method(self):
        self.results, self.model = run_auction(
            [TruthfulBidder(), TruthfulBidder(), TruthfulBidder()],
            n_rounds=10,
            auction_type="second_price",
        )

    def test_runs(self):
        assert len(self.results.model_metrics) == 10

    def test_winner_pays_second_price(self):
        """In second-price, price paid <= winning bid."""
        for wb, pp in zip(
            self.model.winning_bid_history, self.model.price_paid_history, strict=False
        ):
            assert pp <= wb


class TestAggressiveBidderOverbids:
    """Aggressive bidder bids above value."""

    def setup_method(self):
        self.results, self.model = run_auction(
            [AggressiveBidder(), TruthfulBidder()],
            n_rounds=20,
            auction_type="first_price",
        )

    def test_overbidding_detected(self):
        df = self.results.model_metrics
        # Aggressive bidder bids 1.1x value, so at least some overbidding
        assert df["overbidding_rate"].mean() > 0


class TestShadedBidder:
    """Shaded bidder bids below value."""

    def setup_method(self):
        self.results, self.model = run_auction(
            [ShadedBidder(shade_factor=0.5), ShadedBidder(shade_factor=0.5)],
            n_rounds=10,
        )

    def test_runs(self):
        assert len(self.results.model_metrics) == 10

    def test_non_negative_winner_payoffs(self):
        """Shaded bidders bid below value, so winner payoff >= 0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["winner_surplus"] >= 0


class TestMetricsPresent:
    def setup_method(self):
        self.results, _ = run_auction(
            [TruthfulBidder(), ShadedBidder(), AggressiveBidder()], n_rounds=5
        )

    def test_all_metrics(self):
        df = self.results.model_metrics
        for col in [
            "avg_bid",
            "winner_surplus",
            "overbidding_rate",
            "revenue",
            "efficiency",
            "social_welfare",
            "strategy_entropy",
        ]:
            assert col in df.columns


class TestReproducibility:
    def test_deterministic(self):
        _, m1 = run_auction(
            [TruthfulBidder(), BestResponseBidder()], n_rounds=10, seed=99
        )
        _, m2 = run_auction(
            [TruthfulBidder(), BestResponseBidder()], n_rounds=10, seed=99
        )
        p1 = [a.cumulative_payoff for a in sorted(m1.agents, key=lambda a: a.unique_id)]
        p2 = [a.cumulative_payoff for a in sorted(m2.agents, key=lambda a: a.unique_id)]
        assert p1 == p2
