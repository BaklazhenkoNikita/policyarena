"""Tests for LLM brains — uses a mock LLM to avoid real API calls."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

# Skip entire module when optional LLM dependencies are not installed
pytest.importorskip("langchain_core", reason="LLM extras not installed")

from policy_arena.brains.llm.llm_brain import (
    LLMBrain,
    _default_batch_extractor,
    _parse_json_from_response,
)
from policy_arena.core.types import Action, Observation, RoundResult

# --- Battle of the Sexes ---
from policy_arena.games.battle_of_sexes.llm_adapter import (
    BoSDecisionList,
    _bos_observation_formatter,
    bos_llm,
)

# --- Chicken ---
from policy_arena.games.chicken.llm_adapter import (
    CKDecisionList,
    _ck_observation_formatter,
    ck_llm,
)

# --- Tragedy of the Commons ---
from policy_arena.games.commons.llm_adapter import (
    TCDecision,
    _tc_observation_formatter,
    tc_llm,
)
from policy_arena.games.commons.types import TCObservation

# --- El Farol ---
from policy_arena.games.el_farol.llm_adapter import (
    EFDecision,
    _ef_observation_formatter,
    ef_llm,
)
from policy_arena.games.el_farol.types import EFObservation

# --- Hawk-Dove ---
from policy_arena.games.hawk_dove.llm_adapter import (
    HDDecisionList,
    _hd_observation_formatter,
    hd_llm,
)

# --- Minority Game ---
from policy_arena.games.minority_game.llm_adapter import (
    MGDecision,
    _mg_observation_formatter,
    mg_llm,
)
from policy_arena.games.minority_game.types import MGObservation
from policy_arena.games.prisoners_dilemma.llm_adapter import (
    PDDecision,
    PDDecisionList,
    _pd_action_extractor,
    _pd_observation_formatter,
    pd_llm,
)

# --- Public Goods ---
from policy_arena.games.public_goods.llm_adapter import (
    PGDecision,
    _pg_observation_formatter,
    pg_llm,
)
from policy_arena.games.public_goods.types import PGObservation

# --- Stag Hunt ---
from policy_arena.games.stag_hunt.llm_adapter import (
    SHDecisionList,
    _sh_observation_formatter,
    sh_llm,
)

# --- Trust Game ---
from policy_arena.games.trust_game.llm_adapter import (
    TGInvestorDecisionList,
    TGLLMBrain,
    TGTrusteeDecisionList,
    _tg_investor_observation_formatter,
    _tg_trustee_observation_formatter,
    tg_llm_combined,
)
from policy_arena.games.trust_game.types import TGObservation

# --- Ultimatum ---
from policy_arena.games.ultimatum.llm_adapter import (
    UGLLMBrain,
    UGProposerDecisionList,
    UGResponderDecisionList,
    _ug_proposer_observation_formatter,
    _ug_responder_observation_formatter,
    ug_llm_combined,
)
from policy_arena.games.ultimatum.types import UGObservation

# ---------------------------------------------------------------------------
# JSON parser tests
# ---------------------------------------------------------------------------


class TestParseJsonFromResponse:
    def test_plain_json(self):
        text = '{"decisions": [{"opponent": "a", "rationale": "test", "action": 1}]}'
        data = _parse_json_from_response(text)
        assert data["decisions"][0]["action"] == 1

    def test_markdown_fenced_json(self):
        text = '```json\n{"decisions": [{"opponent": "a", "rationale": "test", "action": 0}]}\n```'
        data = _parse_json_from_response(text)
        assert data["decisions"][0]["action"] == 0

    def test_json_with_surrounding_text(self):
        text = 'Here is my decision:\n{"decisions": [{"opponent": "a", "rationale": "test", "action": 1}]}\nDone.'
        data = _parse_json_from_response(text)
        assert data["decisions"][0]["action"] == 1

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError):
            _parse_json_from_response("no json here")


# ---------------------------------------------------------------------------
# Batch action extractor tests
# ---------------------------------------------------------------------------


class TestPDBatchActionExtractor:
    def test_extracts_multiple_actions(self):
        response = PDDecisionList(
            decisions=[
                PDDecision(opponent="a", rationale="cooperate", action=1),
                PDDecision(opponent="b", rationale="defect", action=0),
            ]
        )
        actions = _pd_action_extractor(response, 2)
        assert actions == [Action.COOPERATE, Action.DEFECT]

    def test_pads_missing_decisions(self):
        response = PDDecisionList(
            decisions=[
                PDDecision(opponent="a", rationale="only one", action=0),
            ]
        )
        actions = _pd_action_extractor(response, 3)
        assert len(actions) == 3
        assert actions[0] == Action.DEFECT
        # Padded with COOPERATE
        assert actions[1] == Action.COOPERATE
        assert actions[2] == Action.COOPERATE

    def test_single_opponent(self):
        response = PDDecisionList(
            decisions=[
                PDDecision(opponent="x", rationale="test", action=1),
            ]
        )
        actions = _pd_action_extractor(response, 1)
        assert actions == [Action.COOPERATE]


class TestDefaultBatchExtractor:
    def test_basic_extraction(self):
        response = json.dumps(
            {
                "decisions": [
                    {"opponent": "a", "rationale": "test", "action": 1},
                    {"opponent": "b", "rationale": "test", "action": 0},
                ]
            }
        )
        actions = _default_batch_extractor(response, 2)
        assert actions == [Action.COOPERATE, Action.DEFECT]


# ---------------------------------------------------------------------------
# Batch observation formatter tests
# ---------------------------------------------------------------------------


class TestPDBatchObservationFormatter:
    def test_first_round_single_opponent(self):
        obs = Observation(
            round_number=0,
            extra={"opponent_label": "agent_A", "opponent_brain": "tit_for_tat"},
        )
        text = _pd_observation_formatter([obs])
        assert "Round 1" in text
        assert "Player 1" in text
        assert "first encounter" in text.lower() or "no history" in text.lower()

    def test_does_not_leak_strategy_names(self):
        """Opponent brain/label names must NOT appear in the prompt."""
        obs = Observation(
            round_number=0,
            extra={"opponent_label": "tit_for_tat_0", "opponent_brain": "tit_for_tat"},
        )
        text = _pd_observation_formatter([obs])
        assert "tit_for_tat" not in text
        assert "Player 1" in text

    def test_multiple_opponents(self):
        obs1 = Observation(
            round_number=2,
            my_history=[Action.COOPERATE, Action.DEFECT],
            opponent_history=[Action.DEFECT, Action.COOPERATE],
            extra={"opponent_label": "agent_A", "opponent_brain": "tit_for_tat"},
        )
        obs2 = Observation(
            round_number=2,
            extra={"opponent_label": "agent_B", "opponent_brain": "random"},
        )
        text = _pd_observation_formatter([obs1, obs2])
        assert "Round 3" in text
        assert "2 opponent(s)" in text
        assert "Player 1" in text
        assert "Player 2" in text
        assert "agent_A" not in text
        assert "agent_B" not in text

    def test_uses_compact_cd_notation(self):
        """Should use C/D shorthand instead of full words."""
        obs = Observation(
            round_number=1,
            my_history=[Action.COOPERATE],
            opponent_history=[Action.DEFECT],
            extra={"opponent_label": "opp"},
        )
        text = _pd_observation_formatter([obs])
        assert "C" in text or "D" in text

    def test_shows_cooperation_stats(self):
        obs = Observation(
            my_history=[Action.COOPERATE, Action.COOPERATE, Action.DEFECT],
            opponent_history=[Action.DEFECT, Action.DEFECT, Action.COOPERATE],
            round_number=3,
            extra={"opponent_label": "opp"},
        )
        text = _pd_observation_formatter([obs])
        assert "cooperated" in text.lower()


# ---------------------------------------------------------------------------
# LLMBrain integration tests with mock LLM
# ---------------------------------------------------------------------------


def _make_mock_llm(responses: list[str]) -> MagicMock:
    """Create a mock LLM that returns predetermined string responses."""
    mock = MagicMock()
    response_iter = iter(responses)

    def invoke(messages):
        result = MagicMock()
        result.content = next(response_iter)
        return result

    mock.invoke = invoke
    # with_structured_output returns a new mock that returns Pydantic objects
    mock.with_structured_output = MagicMock(return_value=None)
    return mock


def _make_mock_structured_llm(responses: list[PDDecisionList]) -> MagicMock:
    """Create a mock LLM whose .with_structured_output() returns Pydantic objects."""
    mock = MagicMock()
    response_iter = iter(responses)

    structured_mock = MagicMock()
    structured_mock.invoke = lambda messages: next(response_iter)
    mock.with_structured_output = MagicMock(return_value=structured_mock)

    return mock


def _pd_response(*actions: int) -> PDDecisionList:
    """Create a PDDecisionList for testing."""
    return PDDecisionList(
        decisions=[
            PDDecision(opponent=f"opp_{i}", rationale="test", action=a)
            for i, a in enumerate(actions)
        ]
    )


def _json_response(*actions: int) -> str:
    """Create a valid JSON response string for unstructured testing."""
    decisions = [
        {"opponent": f"opp_{i}", "rationale": "test", "action": a}
        for i, a in enumerate(actions)
    ]
    return json.dumps({"decisions": decisions})


class TestLLMBrain:
    def test_decide_batch_cooperate(self):
        mock_llm = _make_mock_llm([_json_response(1)])
        brain = LLMBrain(llm=mock_llm, brain_name="test_llm")

        obs = Observation(round_number=0, extra={"opponent_label": "opp_0"})
        actions = brain.decide_batch([obs])
        assert actions == [Action.COOPERATE]

    def test_decide_single(self):
        mock_llm = _make_mock_llm([_json_response(0)])
        brain = LLMBrain(llm=mock_llm, brain_name="test_llm")

        obs = Observation(round_number=0, extra={"opponent_label": "opp_0"})
        action = brain.decide(obs)
        assert action == Action.DEFECT

    def test_decide_batch_multiple_opponents(self):
        mock_llm = _make_mock_llm([_json_response(1, 0, 1)])
        brain = LLMBrain(llm=mock_llm, brain_name="test_llm")

        obs_list = [
            Observation(round_number=0, extra={"opponent_label": f"opp_{i}"})
            for i in range(3)
        ]
        actions = brain.decide_batch(obs_list)
        assert actions == [Action.COOPERATE, Action.DEFECT, Action.COOPERATE]

    def test_retry_on_invalid_json(self):
        mock_llm = _make_mock_llm(
            [
                "I'm thinking...",  # invalid first response
                _json_response(1),  # valid retry response
            ]
        )
        brain = LLMBrain(llm=mock_llm, brain_name="test_llm")

        obs = Observation(round_number=0, extra={"opponent_label": "opp_0"})
        actions = brain.decide_batch([obs])
        assert actions == [Action.COOPERATE]

    def test_fallback_on_double_invalid(self):
        mock_llm = _make_mock_llm(
            [
                "hmm...",  # invalid
                "still thinking",  # invalid again
            ]
        )
        brain = LLMBrain(llm=mock_llm, brain_name="test_llm")

        obs = Observation(round_number=0, extra={"opponent_label": "opp_0"})
        actions = brain.decide_batch([obs])
        # Should fallback to COOPERATE for all
        assert actions == [Action.COOPERATE]

    def test_update_adds_to_history(self):
        mock_llm = _make_mock_llm([_json_response(1)])
        brain = LLMBrain(llm=mock_llm, brain_name="test_llm")

        brain.decide(Observation(round_number=0, extra={"opponent_label": "opp"}))

        result = RoundResult(
            action=Action.COOPERATE,
            opponent_action=Action.DEFECT,
            payoff=0.0,
            round_number=0,
        )
        brain.update(result)
        # History: batch prompt, AI response, result message
        assert len(brain._history) == 3

    def test_reset_clears_history(self):
        mock_llm = _make_mock_llm([_json_response(1)])
        brain = LLMBrain(llm=mock_llm, brain_name="test_llm")

        brain.decide(Observation(round_number=0, extra={"opponent_label": "opp"}))
        assert len(brain._history) > 0

        brain.reset()
        assert len(brain._history) == 0

    def test_history_trimming(self):
        responses = [_json_response(1)] * 30
        mock_llm = _make_mock_llm(responses)
        brain = LLMBrain(llm=mock_llm, brain_name="test_llm", max_history=5)

        for i in range(10):
            brain.decide(Observation(round_number=i, extra={"opponent_label": "opp"}))
        # max_history=5 means max 10 messages (5 pairs)
        assert len(brain._history) <= 10

    def test_name_property(self):
        mock_llm = _make_mock_llm([])
        brain = LLMBrain(llm=mock_llm, brain_name="my_agent")
        assert brain.name == "my_agent"


class TestPDLLMAdapter:
    def test_pd_llm_creates_brain_with_structured_output(self):
        """pd_llm should create a brain with structured output enabled."""
        mock_llm = _make_mock_structured_llm([_pd_response(0)])
        brain = pd_llm(model="test-model")
        # Replace LLM internals with mock
        brain._llm = mock_llm
        brain._structured_llm = mock_llm.with_structured_output(PDDecisionList)

        obs = Observation(
            my_history=[Action.COOPERATE],
            opponent_history=[Action.DEFECT],
            round_number=1,
            extra={"opponent_label": "opp", "opponent_brain": "tit_for_tat"},
        )
        action = brain.decide(obs)
        assert action == Action.DEFECT

    def test_pd_llm_has_output_schema(self):
        """pd_llm should set the PDDecisionList output schema."""
        brain = pd_llm(model="test-model")
        assert brain._output_schema is PDDecisionList

    def test_pd_llm_with_persona(self):
        brain = pd_llm(model="test-model", persona="Be aggressive")
        assert "Be aggressive" in brain._persona

    def test_pd_llm_with_characteristics(self):
        brain = pd_llm(
            model="test-model",
            characteristics={
                "personality": "Friendly and trusting",
                "cooperation_bias": 0.8,
            },
        )
        assert "Friendly and trusting" in brain._persona
        assert "cooperation" in brain._persona.lower()

    def test_pd_llm_with_custom_payoff_matrix(self):
        brain = pd_llm(
            model="test-model",
            payoff_matrix={"cc_1": 5.0, "cc_2": 5.0, "dd_1": 0.0, "dd_2": 0.0},
        )
        assert "you get 5" in brain._persona
        assert "you get 0, they get 0" in brain._persona

    def test_pd_llm_default_persona(self):
        brain = pd_llm(model="test-model")
        assert "rational" in brain._persona.lower()
        assert "maximize" in brain._persona.lower()

    def test_pd_llm_system_prompt_has_action_format(self):
        """System prompt should mention 0 = DEFECT, 1 = COOPERATE."""
        brain = pd_llm(model="test-model")
        assert "0 = DEFECT" in brain._persona
        assert "1 = COOPERATE" in brain._persona

    def test_pd_llm_batch_with_multiple_opponents(self):
        """Batch call should produce correct actions for multiple opponents."""
        mock_llm = _make_mock_structured_llm([_pd_response(1, 0)])
        brain = pd_llm(model="test-model")
        brain._llm = mock_llm
        brain._structured_llm = mock_llm.with_structured_output(PDDecisionList)

        obs_list = [
            Observation(
                round_number=0,
                extra={"opponent_label": "agent_A", "opponent_brain": "tit_for_tat"},
            ),
            Observation(
                round_number=0,
                extra={"opponent_label": "agent_B", "opponent_brain": "always_defect"},
            ),
        ]
        actions = brain.decide_batch(obs_list)
        assert actions == [Action.COOPERATE, Action.DEFECT]


class TestStructuredOutput:
    """Tests for the structured output path in LLMBrain."""

    def test_structured_decide_batch(self):
        mock_llm = _make_mock_structured_llm([_pd_response(1, 0)])
        brain = LLMBrain(
            llm=mock_llm,
            output_schema=PDDecisionList,
            batch_action_extractor=_pd_action_extractor,
            brain_name="test_structured",
        )

        obs_list = [
            Observation(round_number=0, extra={"opponent_label": f"opp_{i}"})
            for i in range(2)
        ]
        actions = brain.decide_batch(obs_list)
        assert actions == [Action.COOPERATE, Action.DEFECT]

    def test_structured_fallback_on_error(self):
        """If structured output fails, fallback to COOPERATE."""
        mock_llm = MagicMock()
        structured_mock = MagicMock()
        structured_mock.invoke = MagicMock(side_effect=Exception("model error"))
        mock_llm.with_structured_output = MagicMock(return_value=structured_mock)

        brain = LLMBrain(
            llm=mock_llm,
            output_schema=PDDecisionList,
            batch_action_extractor=_pd_action_extractor,
            brain_name="test_fallback",
        )

        obs = Observation(round_number=0, extra={"opponent_label": "opp_0"})
        actions = brain.decide_batch([obs])
        assert actions == [Action.COOPERATE]


# ---------------------------------------------------------------------------
# El Farol LLM adapter tests
# ---------------------------------------------------------------------------


class TestEFLLMAdapter:
    def test_ef_llm_creates_brain_with_structured_output(self):
        brain = ef_llm(model="test-model")
        assert brain._output_schema is EFDecision

    def test_ef_llm_with_persona(self):
        brain = ef_llm(model="test-model", persona="Be strategic")
        assert "Be strategic" in brain._persona

    def test_ef_llm_default_persona(self):
        brain = ef_llm(model="test-model")
        assert "rational" in brain._persona.lower()

    def test_ef_llm_with_characteristics(self):
        brain = ef_llm(
            model="test-model",
            characteristics={"personality": "Bold gambler"},
        )
        assert "Bold gambler" in brain._persona

    def test_ef_observation_formatter_first_round(self):
        obs = EFObservation(round_number=0, threshold=6, n_agents=10)
        text = _ef_observation_formatter([obs])
        assert "Round 1" in text
        assert "first round" in text.lower() or "no history" in text.lower()

    def test_ef_observation_formatter_with_history(self):
        obs = EFObservation(
            round_number=3,
            threshold=6,
            n_agents=10,
            past_attendance=[4, 7, 5],
            my_past_decisions=[True, False, True],
            my_past_payoffs=[1.0, 0.0, 1.0],
        )
        text = _ef_observation_formatter([obs])
        assert "Round 4" in text
        assert "Go" in text or "Stay" in text

    def test_ef_observation_formatter_no_strategy_leak(self):
        obs = EFObservation(
            round_number=0,
            threshold=6,
            n_agents=10,
        )
        text = _ef_observation_formatter([obs])
        assert "mean_revert" not in text
        assert "always_go" not in text


# ---------------------------------------------------------------------------
# Stag Hunt LLM adapter tests
# ---------------------------------------------------------------------------


class TestSHLLMAdapter:
    def test_sh_llm_creates_brain_with_structured_output(self):
        brain = sh_llm(model="test-model")
        assert brain._output_schema is SHDecisionList

    def test_sh_llm_with_persona(self):
        brain = sh_llm(model="test-model", persona="Always hunt stag")
        assert "Always hunt stag" in brain._persona

    def test_sh_llm_default_persona(self):
        brain = sh_llm(model="test-model")
        assert "rational" in brain._persona.lower()

    def test_sh_llm_with_characteristics(self):
        brain = sh_llm(
            model="test-model",
            characteristics={"personality": "Trusting"},
        )
        assert "Trusting" in brain._persona

    def test_sh_observation_formatter_first_round(self):
        obs = Observation(
            round_number=0,
            extra={"opponent_label": "agent_A", "opponent_brain": "tit_for_tat"},
        )
        text = _sh_observation_formatter([obs])
        assert "Round 1" in text
        assert "first encounter" in text.lower() or "no history" in text.lower()

    def test_sh_observation_formatter_no_strategy_leak(self):
        obs = Observation(
            round_number=0,
            extra={"opponent_label": "pavlov_0", "opponent_brain": "pavlov"},
        )
        text = _sh_observation_formatter([obs])
        assert "pavlov" not in text
        assert "Player 1" in text

    def test_sh_observation_formatter_with_history(self):
        obs = Observation(
            round_number=2,
            my_history=[Action.COOPERATE, Action.DEFECT],
            opponent_history=[Action.COOPERATE, Action.COOPERATE],
            extra={"opponent_label": "opp"},
        )
        text = _sh_observation_formatter([obs])
        assert "Round 3" in text
        assert "S" in text or "H" in text


# ---------------------------------------------------------------------------
# Public Goods LLM adapter tests
# ---------------------------------------------------------------------------


class TestPGLLMAdapter:
    def test_pg_llm_creates_brain_with_structured_output(self):
        brain = pg_llm(model="test-model")
        assert brain._output_schema is PGDecision

    def test_pg_llm_with_persona(self):
        brain = pg_llm(model="test-model", persona="Be generous")
        assert "Be generous" in brain._persona

    def test_pg_llm_default_persona(self):
        brain = pg_llm(model="test-model")
        assert "rational" in brain._persona.lower()

    def test_pg_llm_with_characteristics(self):
        brain = pg_llm(
            model="test-model",
            characteristics={"personality": "Altruistic"},
        )
        assert "Altruistic" in brain._persona

    def test_pg_observation_formatter_first_round(self):
        obs = PGObservation(round_number=0, endowment=20.0, multiplier=1.6, n_players=5)
        text = _pg_observation_formatter([obs])
        assert "Round 1" in text
        assert "first round" in text.lower() or "no history" in text.lower()

    def test_pg_observation_formatter_with_history(self):
        obs = PGObservation(
            round_number=2,
            endowment=20.0,
            multiplier=1.6,
            n_players=5,
            my_past_contributions=[10.0, 15.0],
            group_past_averages=[12.0, 11.0],
            my_past_payoffs=[18.0, 17.0],
        )
        text = _pg_observation_formatter([obs])
        assert "Round 3" in text
        assert "contribution" in text.lower() or "Endowment" in text


# ---------------------------------------------------------------------------
# Tragedy of the Commons LLM adapter tests
# ---------------------------------------------------------------------------


class TestTCLLMAdapter:
    def test_tc_llm_creates_brain_with_structured_output(self):
        brain = tc_llm(model="test-model")
        assert brain._output_schema is TCDecision

    def test_tc_llm_with_persona(self):
        brain = tc_llm(model="test-model", persona="Conserve resources")
        assert "Conserve resources" in brain._persona

    def test_tc_llm_default_persona(self):
        brain = tc_llm(model="test-model")
        assert "rational" in brain._persona.lower()

    def test_tc_llm_with_characteristics(self):
        brain = tc_llm(
            model="test-model",
            characteristics={"personality": "Greedy"},
        )
        assert "Greedy" in brain._persona

    def test_tc_observation_formatter_first_round(self):
        obs = TCObservation(
            round_number=0,
            resource_level=100.0,
            max_resource=100.0,
            growth_rate=1.5,
            n_agents=8,
        )
        text = _tc_observation_formatter([obs])
        assert "Round 1" in text
        assert "resource" in text.lower() or "Resource" in text

    def test_tc_observation_formatter_with_history(self):
        obs = TCObservation(
            round_number=3,
            resource_level=80.0,
            max_resource=100.0,
            growth_rate=1.5,
            n_agents=8,
            resource_history=[100.0, 90.0, 80.0],
            my_past_harvests=[10.0, 8.0, 7.0],
            group_past_total_harvests=[50.0, 45.0, 40.0],
        )
        text = _tc_observation_formatter([obs])
        assert "Round 4" in text
        assert "harvest" in text.lower() or "resource" in text.lower()


# ---------------------------------------------------------------------------
# Minority Game LLM adapter tests
# ---------------------------------------------------------------------------


class TestMGLLMAdapter:
    def test_mg_llm_creates_brain_with_structured_output(self):
        brain = mg_llm(model="test-model")
        assert brain._output_schema is MGDecision

    def test_mg_llm_with_persona(self):
        brain = mg_llm(model="test-model", persona="Be unpredictable")
        assert "Be unpredictable" in brain._persona

    def test_mg_llm_default_persona(self):
        brain = mg_llm(model="test-model")
        assert "rational" in brain._persona.lower()

    def test_mg_llm_with_characteristics(self):
        brain = mg_llm(
            model="test-model",
            characteristics={"personality": "Contrarian thinker"},
        )
        assert "Contrarian thinker" in brain._persona

    def test_mg_observation_formatter_first_round(self):
        obs = MGObservation(round_number=0, n_agents=11)
        text = _mg_observation_formatter([obs])
        assert "Round 1" in text
        assert "11" in text

    def test_mg_observation_formatter_with_history(self):
        obs = MGObservation(
            round_number=3,
            n_agents=11,
            past_winning_sides=["A", "B", "A"],
            past_a_counts=[6, 4, 7],
            my_past_choices=[True, False, True],
            my_past_payoffs=[-1.0, 1.0, -1.0],
        )
        text = _mg_observation_formatter([obs])
        assert "Round 4" in text
        assert "A" in text and "B" in text


# ---------------------------------------------------------------------------
# Battle of the Sexes LLM adapter tests
# ---------------------------------------------------------------------------


class TestBoSLLMAdapter:
    def test_bos_llm_creates_brain_with_structured_output(self):
        brain = bos_llm(model="test-model")
        assert brain._output_schema is BoSDecisionList

    def test_bos_llm_with_persona(self):
        brain = bos_llm(model="test-model", persona="Prefer Option A")
        assert "Prefer Option A" in brain._persona

    def test_bos_llm_default_persona(self):
        brain = bos_llm(model="test-model")
        assert "rational" in brain._persona.lower()

    def test_bos_llm_with_characteristics(self):
        brain = bos_llm(
            model="test-model",
            characteristics={"personality": "Stubborn negotiator"},
        )
        assert "Stubborn negotiator" in brain._persona

    def test_bos_observation_formatter_first_round(self):
        obs = Observation(
            round_number=0,
            extra={"opponent_label": "agent_A", "opponent_brain": "always_a"},
        )
        text = _bos_observation_formatter([obs])
        assert "Round 1" in text
        assert "first encounter" in text.lower() or "no history" in text.lower()

    def test_bos_observation_formatter_no_strategy_leak(self):
        obs = Observation(
            round_number=0,
            extra={"opponent_label": "always_a_0", "opponent_brain": "always_a"},
        )
        text = _bos_observation_formatter([obs])
        assert "always_a" not in text
        assert "Player 1" in text

    def test_bos_observation_formatter_multiple_opponents(self):
        obs1 = Observation(
            round_number=2,
            my_history=[Action.COOPERATE, Action.DEFECT],
            opponent_history=[Action.COOPERATE, Action.COOPERATE],
            extra={"opponent_label": "agent_A"},
        )
        obs2 = Observation(
            round_number=2,
            extra={"opponent_label": "agent_B"},
        )
        text = _bos_observation_formatter([obs1, obs2])
        assert "2 opponent(s)" in text
        assert "Player 1" in text
        assert "Player 2" in text
        assert "agent_A" not in text
        assert "agent_B" not in text


# ---------------------------------------------------------------------------
# Hawk-Dove LLM adapter tests
# ---------------------------------------------------------------------------


class TestHDLLMAdapter:
    def test_hd_llm_creates_brain_with_structured_output(self):
        brain = hd_llm(model="test-model")
        assert brain._output_schema is HDDecisionList

    def test_hd_llm_with_persona(self):
        brain = hd_llm(model="test-model", persona="Play dove always")
        assert "Play dove always" in brain._persona

    def test_hd_llm_default_persona(self):
        brain = hd_llm(model="test-model")
        assert "rational" in brain._persona.lower()

    def test_hd_llm_with_characteristics(self):
        brain = hd_llm(
            model="test-model",
            characteristics={"personality": "Aggressive hawk"},
        )
        assert "Aggressive hawk" in brain._persona

    def test_hd_observation_formatter_first_round(self):
        obs = Observation(
            round_number=0,
            extra={"opponent_label": "agent_A", "opponent_brain": "hawk_strat"},
        )
        text = _hd_observation_formatter([obs])
        assert "Round 1" in text
        assert "first encounter" in text.lower() or "no history" in text.lower()

    def test_hd_observation_formatter_no_strategy_leak(self):
        obs = Observation(
            round_number=0,
            extra={"opponent_label": "hawk_strat_0", "opponent_brain": "hawk_strat"},
        )
        text = _hd_observation_formatter([obs])
        assert "hawk_strat" not in text
        assert "Player 1" in text

    def test_hd_observation_formatter_with_history(self):
        obs = Observation(
            round_number=2,
            my_history=[Action.COOPERATE, Action.DEFECT],
            opponent_history=[Action.DEFECT, Action.COOPERATE],
            extra={"opponent_label": "opp"},
        )
        text = _hd_observation_formatter([obs])
        assert "Round 3" in text
        assert "D" in text or "H" in text


# ---------------------------------------------------------------------------
# Chicken LLM adapter tests
# ---------------------------------------------------------------------------


class TestCKLLMAdapter:
    def test_ck_llm_creates_brain_with_structured_output(self):
        brain = ck_llm(model="test-model")
        assert brain._output_schema is CKDecisionList

    def test_ck_llm_with_persona(self):
        brain = ck_llm(model="test-model", persona="Swerve cautiously")
        assert "Swerve cautiously" in brain._persona

    def test_ck_llm_default_persona(self):
        brain = ck_llm(model="test-model")
        assert "rational" in brain._persona.lower()

    def test_ck_llm_with_characteristics(self):
        brain = ck_llm(
            model="test-model",
            characteristics={"personality": "Daredevil"},
        )
        assert "Daredevil" in brain._persona

    def test_ck_observation_formatter_first_round(self):
        obs = Observation(
            round_number=0,
            extra={"opponent_label": "agent_A", "opponent_brain": "always_straight"},
        )
        text = _ck_observation_formatter([obs])
        assert "Round 1" in text
        assert "first encounter" in text.lower() or "no history" in text.lower()

    def test_ck_observation_formatter_no_strategy_leak(self):
        obs = Observation(
            round_number=0,
            extra={
                "opponent_label": "always_straight_0",
                "opponent_brain": "always_straight",
            },
        )
        text = _ck_observation_formatter([obs])
        assert "always_straight" not in text
        assert "Player 1" in text

    def test_ck_observation_formatter_with_history(self):
        obs = Observation(
            round_number=2,
            my_history=[Action.COOPERATE, Action.DEFECT],
            opponent_history=[Action.DEFECT, Action.DEFECT],
            extra={"opponent_label": "opp"},
        )
        text = _ck_observation_formatter([obs])
        assert "Round 3" in text
        assert "Sw" in text or "St" in text


# ---------------------------------------------------------------------------
# Trust Game LLM adapter tests
# ---------------------------------------------------------------------------


class TestTGLLMAdapter:
    def test_tg_llm_combined_creates_dual_brain(self):
        brain = tg_llm_combined(model="test-model")
        assert isinstance(brain, TGLLMBrain)
        assert hasattr(brain, "_investor")
        assert hasattr(brain, "_trustee")

    def test_tg_llm_combined_has_name(self):
        brain = tg_llm_combined(model="test-model")
        assert brain.name == "llm(ollama/test-model)"

    def test_tg_llm_combined_reset_clears_both(self):
        brain = tg_llm_combined(model="test-model")
        brain._investor._history = [{"role": "user", "content": "test"}]
        brain._trustee._history = [{"role": "user", "content": "test"}]
        brain.reset()
        assert len(brain._investor._history) == 0
        assert len(brain._trustee._history) == 0

    def test_tg_llm_combined_with_persona(self):
        brain = tg_llm_combined(model="test-model", persona="Be trusting")
        assert "Be trusting" in brain._investor._persona
        assert "Be trusting" in brain._trustee._persona

    def test_tg_llm_combined_default_persona(self):
        brain = tg_llm_combined(model="test-model")
        assert "rational" in brain._investor._persona.lower()
        assert "rational" in brain._trustee._persona.lower()

    def test_tg_llm_combined_with_characteristics(self):
        brain = tg_llm_combined(
            model="test-model",
            characteristics={"personality": "Fair dealer"},
        )
        assert "Fair dealer" in brain._investor._persona
        assert "Fair dealer" in brain._trustee._persona

    def test_tg_llm_investor_schema(self):
        brain = tg_llm_combined(model="test-model")
        assert brain._investor._output_schema is TGInvestorDecisionList

    def test_tg_llm_trustee_schema(self):
        brain = tg_llm_combined(model="test-model")
        assert brain._trustee._output_schema is TGTrusteeDecisionList

    def test_tg_investor_observation_formatter_first_round(self):
        obs = TGObservation(
            role="investor", round_number=0, endowment=10.0, multiplier=3.0
        )
        text = _tg_investor_observation_formatter([obs])
        assert "Round 1" in text
        assert "INVESTOR" in text
        assert "first interaction" in text.lower() or "no history" in text.lower()

    def test_tg_trustee_observation_formatter(self):
        obs = TGObservation(
            role="trustee",
            round_number=1,
            endowment=10.0,
            multiplier=3.0,
            amount_received=15.0,
        )
        text = _tg_trustee_observation_formatter([obs])
        assert "Round 2" in text
        assert "TRUSTEE" in text
        assert "15" in text


# ---------------------------------------------------------------------------
# Ultimatum Game LLM adapter tests
# ---------------------------------------------------------------------------


class TestUGLLMAdapter:
    def test_ug_llm_combined_creates_dual_brain(self):
        brain = ug_llm_combined(model="test-model")
        assert isinstance(brain, UGLLMBrain)
        assert hasattr(brain, "_proposer")
        assert hasattr(brain, "_responder")

    def test_ug_llm_combined_has_name(self):
        brain = ug_llm_combined(model="test-model")
        assert brain.name == "llm(ollama/test-model)"

    def test_ug_llm_combined_reset_clears_both(self):
        brain = ug_llm_combined(model="test-model")
        brain._proposer._history = [{"role": "user", "content": "test"}]
        brain._responder._history = [{"role": "user", "content": "test"}]
        brain.reset()
        assert len(brain._proposer._history) == 0
        assert len(brain._responder._history) == 0

    def test_ug_llm_combined_with_persona(self):
        brain = ug_llm_combined(model="test-model", persona="Be fair")
        assert "Be fair" in brain._proposer._persona
        assert "Be fair" in brain._responder._persona

    def test_ug_llm_combined_default_persona(self):
        brain = ug_llm_combined(model="test-model")
        assert "rational" in brain._proposer._persona.lower()
        assert "rational" in brain._responder._persona.lower()

    def test_ug_llm_combined_with_characteristics(self):
        brain = ug_llm_combined(
            model="test-model",
            characteristics={"personality": "Shrewd negotiator"},
        )
        assert "Shrewd negotiator" in brain._proposer._persona
        assert "Shrewd negotiator" in brain._responder._persona

    def test_ug_llm_proposer_schema(self):
        brain = ug_llm_combined(model="test-model")
        assert brain._proposer._output_schema is UGProposerDecisionList

    def test_ug_llm_responder_schema(self):
        brain = ug_llm_combined(model="test-model")
        assert brain._responder._output_schema is UGResponderDecisionList

    def test_ug_proposer_observation_formatter_first_round(self):
        obs = UGObservation(role="proposer", stake=100.0, round_number=0)
        text = _ug_proposer_observation_formatter([obs])
        assert "Round 1" in text
        assert "PROPOSER" in text
        assert "first interaction" in text.lower() or "no history" in text.lower()

    def test_ug_responder_observation_formatter(self):
        obs = UGObservation(role="responder", stake=100.0, round_number=1, offer=40.0)
        text = _ug_responder_observation_formatter([obs])
        assert "Round 2" in text
        assert "RESPONDER" in text
        assert "40" in text

    def test_ug_proposer_observation_formatter_with_history(self):
        obs = UGObservation(
            role="proposer",
            stake=100.0,
            round_number=3,
            my_past_offers_made=[50.0, 40.0, 30.0],
            my_past_responses=[True, True, False],
        )
        text = _ug_proposer_observation_formatter([obs])
        assert "Round 4" in text
        assert "past offers" in text.lower() or "50" in text
