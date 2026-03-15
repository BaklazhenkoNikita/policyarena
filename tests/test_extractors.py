"""Tests for core.extractors — state extraction from game models."""

from policy_arena.core.engine import Engine
from policy_arena.core.extractors import (
    extract_agent_states,
    extract_game_data,
    extract_model_metrics,
)
from policy_arena.core.scenario import Scenario
from policy_arena.games.auction.brains import TruthfulBidder
from policy_arena.games.auction.model import AuctionModel
from policy_arena.games.battle_of_sexes.brains import AlwaysA as BosAlwaysA
from policy_arena.games.battle_of_sexes.brains import AlwaysB as BosAlwaysB
from policy_arena.games.battle_of_sexes.model import BattleOfSexesModel
from policy_arena.games.chicken.brains import AlwaysStraight, AlwaysSwerve
from policy_arena.games.chicken.model import ChickenModel

# Import game models and brains for integration-style extraction tests
from policy_arena.games.commons.brains import Greedy, Sustainable
from policy_arena.games.commons.model import CommonsModel
from policy_arena.games.cournot.brains import NashEquilibrium as CournotNE
from policy_arena.games.cournot.model import CournotModel
from policy_arena.games.el_farol.brains import AlwaysAttend, NeverAttend
from policy_arena.games.el_farol.model import ElFarolModel
from policy_arena.games.hawk_dove.brains import AlwaysDove, AlwaysHawk
from policy_arena.games.hawk_dove.model import HawkDoveModel
from policy_arena.games.info_cascade.brains import SignalFollower
from policy_arena.games.info_cascade.model import CascadeModel
from policy_arena.games.lobbying.brains import NashEquilibrium as LobbyingNE
from policy_arena.games.lobbying.model import LobbyingModel
from policy_arena.games.minority_game.brains import AlwaysA, AlwaysB
from policy_arena.games.minority_game.model import MinorityGameModel
from policy_arena.games.network_formation.brains import FullyConnected
from policy_arena.games.network_formation.model import NetworkModel
from policy_arena.games.prisoners_dilemma.model import PrisonersDilemmaModel
from policy_arena.games.public_goods.brains import FreeRider, FullContributor
from policy_arena.games.public_goods.model import PublicGoodsModel
from policy_arena.games.schelling.brains import TolerantBrain
from policy_arena.games.schelling.model import SchellingModel
from policy_arena.games.sir.brains import NeverIsolate
from policy_arena.games.sir.model import SIRModel

# Additional game imports for extraction coverage
from policy_arena.games.stag_hunt.brains import AlwaysHare, AlwaysStag
from policy_arena.games.stag_hunt.model import StagHuntModel
from policy_arena.games.trust_game.brains import FairPlayer as TrustFairPlayer
from policy_arena.games.trust_game.brains import FullTrust
from policy_arena.games.trust_game.model import TrustGameModel
from policy_arena.games.ultimatum.brains import FairPlayer as UltimatumFairPlayer
from policy_arena.games.ultimatum.brains import GreedyPlayer
from policy_arena.games.ultimatum.model import UltimatumModel
from policy_arena.games.voting.brains import SincereVoter
from policy_arena.games.voting.model import VotingModel


def _run_and_extract(world_class, world_params, game_id, n_rounds=3, seed=42):
    """Run a game and extract states after completion."""
    scenario = Scenario(
        world_class=world_class,
        world_params=world_params,
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    model = results.extra["model"]
    agents = extract_agent_states(model, game_id)
    metrics = extract_model_metrics(model, game_id)
    game_data = extract_game_data(model, game_id)
    return agents, metrics, game_data, model


class TestPrisonersDilemmaExtraction:
    def setup_method(self):
        from policy_arena.brains.rule_based.always_cooperate import AlwaysCooperate
        from policy_arena.brains.rule_based.always_defect import AlwaysDefect

        self.agents, self.metrics, self.game_data, _ = _run_and_extract(
            PrisonersDilemmaModel,
            {"brains": [AlwaysCooperate(), AlwaysDefect()], "n_rounds": 3},
            "prisoners_dilemma",
        )

    def test_agent_count(self):
        assert len(self.agents) == 2

    def test_agent_fields(self):
        for a in self.agents:
            assert "id" in a
            assert "label" in a
            assert "brain_name" in a
            assert "cumulative_payoff" in a
            assert "extra" in a

    def test_action_present(self):
        actions = [a["extra"]["action"] for a in self.agents]
        assert "cooperate" in actions or "defect" in actions

    def test_matchups_present(self):
        for a in self.agents:
            assert "matchups" in a["extra"]
            for m in a["extra"]["matchups"]:
                assert "opponent_id" in m
                assert "action" in m
                assert "payoff" in m


class TestElFarolExtraction:
    def setup_method(self):
        self.agents, self.metrics, self.game_data, _ = _run_and_extract(
            ElFarolModel,
            {"brains": [AlwaysAttend()] * 3 + [NeverAttend()] * 2, "n_rounds": 3},
            "el_farol",
        )

    def test_attended_field(self):
        for a in self.agents:
            assert "attended" in a["extra"]

    def test_action_field(self):
        actions = {a["extra"]["action"] for a in self.agents}
        assert actions <= {"attend", "stay", None}

    def test_game_data_has_history(self):
        assert "attendance_history" in self.game_data
        assert "threshold" in self.game_data


class TestPublicGoodsExtraction:
    def setup_method(self):
        self.agents, self.metrics, self.game_data, _ = _run_and_extract(
            PublicGoodsModel,
            {"brains": [FullContributor(), FreeRider()], "n_rounds": 3},
            "public_goods",
        )

    def test_contribution_field(self):
        for a in self.agents:
            assert "last_contribution" in a["extra"]

    def test_game_data(self):
        assert "group_avg_history" in self.game_data


class TestCommonsExtraction:
    def setup_method(self):
        self.agents, self.metrics, self.game_data, _ = _run_and_extract(
            CommonsModel,
            {"brains": [Greedy(), Sustainable()], "n_rounds": 3},
            "commons",
        )

    def test_harvest_fields(self):
        for a in self.agents:
            assert "last_harvest" in a["extra"]
            assert "sustainable_harvest" in a["extra"]
            assert "harvest_cap" in a["extra"]
            assert "overusing" in a["extra"]


class TestUltimatumExtraction:
    def setup_method(self):
        self.agents, self.metrics, self.game_data, _ = _run_and_extract(
            UltimatumModel,
            {"brains": [UltimatumFairPlayer(), GreedyPlayer()], "n_rounds": 3},
            "ultimatum",
        )

    def test_role_fields(self):
        for a in self.agents:
            extra = a["extra"]
            assert "last_role" in extra
            assert "last_offer" in extra


class TestTrustGameExtraction:
    def setup_method(self):
        self.agents, self.metrics, self.game_data, _ = _run_and_extract(
            TrustGameModel,
            {"brains": [FullTrust(), TrustFairPlayer()], "n_rounds": 3},
            "trust_game",
        )

    def test_role_fields(self):
        for a in self.agents:
            extra = a["extra"]
            assert "last_role" in extra
            assert "last_investment" in extra
            assert "last_return" in extra


class TestMinorityGameExtraction:
    def setup_method(self):
        self.agents, self.metrics, self.game_data, _ = _run_and_extract(
            MinorityGameModel,
            {"brains": [AlwaysA(), AlwaysB(), AlwaysA()], "n_rounds": 3},
            "minority_game",
        )

    def test_action_field(self):
        actions = {a["extra"]["action"] for a in self.agents}
        assert actions <= {"A", "B", None}


class TestSchellingExtraction:
    def setup_method(self):
        self.agents, self.metrics, self.game_data, _ = _run_and_extract(
            SchellingModel,
            {
                "brains": [TolerantBrain()] * 10,
                "n_rounds": 3,
                "width": 5,
                "height": 5,
            },
            "schelling",
        )

    def test_agent_type_field(self):
        for a in self.agents:
            assert "agent_type" in a["extra"]

    def test_game_data(self):
        assert "width" in self.game_data
        assert "height" in self.game_data
        assert "type_counts" in self.game_data
        assert "type_stats" in self.game_data


class TestStagHuntExtraction:
    def setup_method(self):
        self.agents, self.metrics, self.game_data, _ = _run_and_extract(
            StagHuntModel,
            {"brains": [AlwaysStag(), AlwaysHare()], "n_rounds": 3},
            "stag_hunt",
        )

    def test_action_field(self):
        actions = {a["extra"]["action"] for a in self.agents}
        assert actions <= {"stag", "hare", "mixed", None}

    def test_matchups(self):
        for a in self.agents:
            assert "matchups" in a["extra"]


class TestBattleOfSexesExtraction:
    def setup_method(self):
        self.agents, self.metrics, self.game_data, _ = _run_and_extract(
            BattleOfSexesModel,
            {"brains": [BosAlwaysA(), BosAlwaysB()], "n_rounds": 3},
            "battle_of_sexes",
        )

    def test_action_field(self):
        actions = {a["extra"]["action"] for a in self.agents}
        assert actions <= {"option_a", "option_b", "mixed", None}

    def test_matchups(self):
        for a in self.agents:
            assert "matchups" in a["extra"]


class TestHawkDoveExtraction:
    def setup_method(self):
        self.agents, self.metrics, self.game_data, _ = _run_and_extract(
            HawkDoveModel,
            {"brains": [AlwaysDove(), AlwaysHawk()], "n_rounds": 3},
            "hawk_dove",
        )

    def test_action_field(self):
        actions = {a["extra"]["action"] for a in self.agents}
        assert actions <= {"dove", "hawk", "mixed", None}

    def test_matchups(self):
        for a in self.agents:
            assert "matchups" in a["extra"]


class TestChickenExtraction:
    def setup_method(self):
        self.agents, self.metrics, self.game_data, _ = _run_and_extract(
            ChickenModel,
            {"brains": [AlwaysSwerve(), AlwaysStraight()], "n_rounds": 3},
            "chicken",
        )

    def test_action_field(self):
        actions = {a["extra"]["action"] for a in self.agents}
        assert actions <= {"swerve", "straight", "mixed", None}

    def test_matchups(self):
        for a in self.agents:
            assert "matchups" in a["extra"]


class TestSIRExtraction:
    def setup_method(self):
        self.agents, self.metrics, self.game_data, _ = _run_and_extract(
            SIRModel,
            {"brains": [NeverIsolate()] * 5, "n_rounds": 3},
            "sir",
        )

    def test_health_state_field(self):
        for a in self.agents:
            assert "health_state" in a["extra"]

    def test_action_field(self):
        for a in self.agents:
            assert a["extra"]["action"] in ("isolating", "participating")

    def test_game_data(self):
        assert "beta" in self.game_data
        assert "gamma" in self.game_data
        assert "susceptible_count" in self.game_data
        assert "infected_count" in self.game_data
        assert "total_agents" in self.game_data
        assert "nodes" in self.game_data
        assert "edges" in self.game_data


class TestVotingExtraction:
    def setup_method(self):
        self.agents, _, _, _ = _run_and_extract(
            VotingModel,
            {
                "brains": [SincereVoter(), SincereVoter()],
                "n_rounds": 3,
                "ideal_points": [20.0, 80.0],
            },
            "voting",
        )

    def test_no_extra_crash(self):
        """Voting doesn't have a specific extractor path but should not crash."""
        assert len(self.agents) == 2


class TestCournotExtraction:
    def setup_method(self):
        self.agents, _, _, _ = _run_and_extract(
            CournotModel,
            {"brains": [CournotNE(), CournotNE()], "n_rounds": 3},
            "cournot",
        )

    def test_no_crash(self):
        assert len(self.agents) == 2


class TestAuctionExtraction:
    def setup_method(self):
        self.agents, _, _, _ = _run_and_extract(
            AuctionModel,
            {"brains": [TruthfulBidder(), TruthfulBidder()], "n_rounds": 3},
            "auction",
        )

    def test_no_crash(self):
        assert len(self.agents) == 2


class TestLobbyingExtraction:
    def setup_method(self):
        self.agents, _, _, _ = _run_and_extract(
            LobbyingModel,
            {"brains": [LobbyingNE(), LobbyingNE()], "n_rounds": 3},
            "lobbying",
        )

    def test_no_crash(self):
        assert len(self.agents) == 2


class TestInfoCascadeExtraction:
    def setup_method(self):
        self.agents, _, _, _ = _run_and_extract(
            CascadeModel,
            {"brains": [SignalFollower()] * 3, "n_rounds": 3},
            "info_cascade",
        )

    def test_no_crash(self):
        assert len(self.agents) == 3


class TestNetworkFormationExtraction:
    def setup_method(self):
        self.agents, _, _, _ = _run_and_extract(
            NetworkModel,
            {"brains": [FullyConnected()] * 3, "n_rounds": 3},
            "network_formation",
        )

    def test_no_crash(self):
        assert len(self.agents) == 3


class TestModelMetrics:
    def test_metrics_dict(self):
        _, metrics, _, _ = _run_and_extract(
            ElFarolModel,
            {"brains": [AlwaysAttend()] * 5, "n_rounds": 3},
            "el_farol",
        )
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        for _k, v in metrics.items():
            assert isinstance(v, float)
