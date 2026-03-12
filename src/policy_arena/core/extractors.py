"""Extract per-round state from Mesa models into plain dicts.

These helpers are used by both the CLI and the API layer.  They return
plain dicts (not Pydantic models) so that the core package has no
dependency on the API schemas.
"""

from __future__ import annotations

import json
from typing import Any

from policy_arena.core.types import Action
from policy_arena.games.sir.types import HealthState


def _is_llm_brain(brain: object) -> bool:
    """Check if a brain is an LLMBrain without requiring LLM deps."""
    try:
        from policy_arena.brains.llm.llm_brain import LLMBrain
    except ImportError:
        return False
    return isinstance(brain, LLMBrain)


def extract_agent_states(model: Any, game_id: str) -> list[dict[str, Any]]:
    """Return a list of dicts with the same shape as the API ``AgentState``."""
    agents = []
    for agent in model.agents:
        extra: dict[str, Any] = {}
        if game_id == "prisoners_dilemma":
            coop_rate = getattr(agent, "cooperated", None)
            extra["cooperated"] = coop_rate
            if coop_rate is None:
                extra["action"] = None
            elif coop_rate == 1.0:
                extra["action"] = "cooperate"
            elif coop_rate == 0.0:
                extra["action"] = "defect"
            else:
                extra["action"] = "mixed"
            opponent_results = getattr(agent, "_round_opponent_results", {})
            agent_map = {a.unique_id: a for a in model.agents}
            matchups = []
            for opp_id, (my_act, opp_act, payoff) in opponent_results.items():
                opp_agent = agent_map.get(opp_id)
                matchups.append(
                    {
                        "opponent_id": opp_id,
                        "opponent_label": getattr(opp_agent, "label", str(opp_id)),
                        "opponent_brain": getattr(opp_agent, "brain_name", "unknown"),
                        "action": my_act.value,
                        "opponent_action": opp_act.value,
                        "payoff": payoff,
                    }
                )
            extra["matchups"] = matchups
        elif game_id == "stag_hunt":
            coop_rate = getattr(agent, "cooperated", None)
            extra["cooperated"] = coop_rate
            if coop_rate is None:
                extra["action"] = None
            elif coop_rate == 1.0:
                extra["action"] = "stag"
            elif coop_rate == 0.0:
                extra["action"] = "hare"
            else:
                extra["action"] = "mixed"
            opponent_results = getattr(agent, "_round_opponent_results", {})
            agent_map_sh = {a.unique_id: a for a in model.agents}
            matchups_sh = []
            for opp_id, (my_act, opp_act, payoff) in opponent_results.items():
                opp_agent = agent_map_sh.get(opp_id)
                matchups_sh.append(
                    {
                        "opponent_id": opp_id,
                        "opponent_label": getattr(opp_agent, "label", str(opp_id)),
                        "opponent_brain": getattr(opp_agent, "brain_name", "unknown"),
                        "action": "stag" if my_act == Action.COOPERATE else "hare",
                        "opponent_action": "stag"
                        if opp_act == Action.COOPERATE
                        else "hare",
                        "payoff": payoff,
                    }
                )
            extra["matchups"] = matchups_sh
        elif game_id == "battle_of_sexes":
            coop_rate = getattr(agent, "cooperated", None)
            extra["cooperated"] = coop_rate
            if coop_rate is None:
                extra["action"] = None
            elif coop_rate == 1.0:
                extra["action"] = "option_a"
            elif coop_rate == 0.0:
                extra["action"] = "option_b"
            else:
                extra["action"] = "mixed"
            opponent_results = getattr(agent, "_round_opponent_results", {})
            agent_map_bos = {a.unique_id: a for a in model.agents}
            matchups_bos = []
            for opp_id, (my_act, opp_act, payoff) in opponent_results.items():
                opp_agent = agent_map_bos.get(opp_id)
                matchups_bos.append(
                    {
                        "opponent_id": opp_id,
                        "opponent_label": getattr(opp_agent, "label", str(opp_id)),
                        "opponent_brain": getattr(opp_agent, "brain_name", "unknown"),
                        "action": "A" if my_act == Action.COOPERATE else "B",
                        "opponent_action": "A" if opp_act == Action.COOPERATE else "B",
                        "payoff": payoff,
                    }
                )
            extra["matchups"] = matchups_bos
        elif game_id == "hawk_dove":
            coop_rate = getattr(agent, "cooperated", None)
            extra["cooperated"] = coop_rate
            if coop_rate is None:
                extra["action"] = None
            elif coop_rate == 1.0:
                extra["action"] = "dove"
            elif coop_rate == 0.0:
                extra["action"] = "hawk"
            else:
                extra["action"] = "mixed"
            opponent_results = getattr(agent, "_round_opponent_results", {})
            agent_map_hd = {a.unique_id: a for a in model.agents}
            matchups_hd = []
            for opp_id, (my_act, opp_act, payoff) in opponent_results.items():
                opp_agent = agent_map_hd.get(opp_id)
                matchups_hd.append(
                    {
                        "opponent_id": opp_id,
                        "opponent_label": getattr(opp_agent, "label", str(opp_id)),
                        "opponent_brain": getattr(opp_agent, "brain_name", "unknown"),
                        "action": "dove" if my_act == Action.COOPERATE else "hawk",
                        "opponent_action": "dove"
                        if opp_act == Action.COOPERATE
                        else "hawk",
                        "payoff": payoff,
                    }
                )
            extra["matchups"] = matchups_hd
        elif game_id == "chicken":
            coop_rate = getattr(agent, "cooperated", None)
            extra["cooperated"] = coop_rate
            if coop_rate is None:
                extra["action"] = None
            elif coop_rate == 1.0:
                extra["action"] = "swerve"
            elif coop_rate == 0.0:
                extra["action"] = "straight"
            else:
                extra["action"] = "mixed"
            opponent_results = getattr(agent, "_round_opponent_results", {})
            agent_map_ck = {a.unique_id: a for a in model.agents}
            matchups_ck = []
            for opp_id, (my_act, opp_act, payoff) in opponent_results.items():
                opp_agent = agent_map_ck.get(opp_id)
                matchups_ck.append(
                    {
                        "opponent_id": opp_id,
                        "opponent_label": getattr(opp_agent, "label", str(opp_id)),
                        "opponent_brain": getattr(opp_agent, "brain_name", "unknown"),
                        "action": "swerve"
                        if my_act == Action.COOPERATE
                        else "straight",
                        "opponent_action": "swerve"
                        if opp_act == Action.COOPERATE
                        else "straight",
                        "payoff": payoff,
                    }
                )
            extra["matchups"] = matchups_ck
        elif game_id == "trust_game":
            last_role = getattr(agent, "_last_role", None)
            last_investment = getattr(agent, "_last_investment", None)
            last_return = getattr(agent, "_last_return", None)
            extra["last_role"] = last_role
            extra["last_investment"] = last_investment
            extra["last_return"] = last_return
            investments = getattr(agent, "_investments_made", [])
            returns = getattr(agent, "_returns_made", [])
            extra["avg_investment"] = (
                sum(investments) / len(investments) if investments else None
            )
            extra["avg_return"] = sum(returns) / len(returns) if returns else None
            if last_role == "investor":
                extra["action"] = (
                    f"invested {last_investment:.1f}"
                    if last_investment is not None
                    else None
                )
            elif last_role == "trustee":
                extra["action"] = (
                    f"returned {last_return:.1f}" if last_return is not None else None
                )
        elif game_id == "el_farol":
            extra["attended"] = getattr(agent, "attended", None)
            extra["action"] = (
                "attend"
                if extra["attended"]
                else "stay"
                if extra["attended"] is not None
                else None
            )
        elif game_id == "public_goods":
            extra["last_contribution"] = getattr(agent, "last_contribution", None)
            extra["action"] = (
                f"{extra['last_contribution']:.1f}"
                if extra["last_contribution"] is not None
                else None
            )
        elif game_id == "ultimatum":
            last_role = getattr(agent, "_last_role", None)
            last_offer = getattr(agent, "_last_offer", None)
            last_accepted = getattr(agent, "_last_accepted", None)
            offers_made = getattr(agent, "_offers_made", [])
            responses_given = getattr(agent, "_responses_given", [])
            extra["last_role"] = last_role
            extra["last_offer"] = last_offer
            extra["last_accepted"] = last_accepted
            extra["avg_offer"] = (
                sum(offers_made) / len(offers_made) if offers_made else None
            )
            extra["accept_pct"] = (
                sum(responses_given) / len(responses_given) if responses_given else None
            )
            if last_role == "proposer":
                extra["action"] = (
                    f"offered {last_offer:.0f}" if last_offer is not None else None
                )
            elif last_role == "responder":
                extra["action"] = (
                    "accepted"
                    if last_accepted
                    else "rejected"
                    if last_accepted is not None
                    else None
                )
        elif game_id == "schelling":
            extra["agent_type"] = getattr(agent, "agent_type", None)
            extra["happy"] = getattr(agent, "happy", None)
            extra["moved"] = getattr(agent, "moved", None)
            pos = getattr(agent, "pos", None)
            if pos is not None:
                extra["x"] = pos[0]
                extra["y"] = pos[1]
            if extra["moved"]:
                extra["action"] = "moved"
            elif extra["moved"] is False:
                extra["action"] = "stayed"
            if hasattr(agent, "get_neighborhood_info") and pos is not None:
                frac_same, frac_diff, n_occ, n_empty, frac_per_type = (
                    agent.get_neighborhood_info()
                )
                extra["fraction_same"] = round(frac_same, 3)
                extra["fraction_different"] = round(frac_diff, 3)
                extra["n_neighbors"] = n_occ
                extra["n_empty_neighbors"] = n_empty
                extra["fraction_per_type"] = {
                    str(k): round(v, 3) for k, v in frac_per_type.items()
                }
            extra["tolerance"] = getattr(model, "tolerance", 0.0)
            gt = getattr(agent, "group_tolerances", {})
            if gt:
                extra["group_tolerances"] = {str(k): v for k, v in gt.items()}
        elif game_id == "sir":
            health = getattr(agent, "health_state", None)
            extra["health_state"] = health.value if health else None
            extra["isolated"] = getattr(agent, "isolated", None)
            extra["days_infected"] = getattr(agent, "days_infected", 0)
            extra["vaccinated"] = getattr(agent, "vaccinated", False)
            extra["immunity"] = round(getattr(agent, "immunity", 0.0), 3)
            if extra["isolated"]:
                extra["action"] = "isolating"
            else:
                extra["action"] = "participating"
        elif game_id == "commons":
            extra["last_harvest"] = getattr(agent, "last_harvest", None)
            extra["action"] = (
                f"{extra['last_harvest']:.1f}"
                if extra["last_harvest"] is not None
                else None
            )
            n_agents = len(list(model.agents))
            resource_before = (
                model.resource_history[-1]
                if model.resource_history
                else model.max_resource
            )
            sustainable_total = resource_before * (model.growth_rate - 1)
            extra["sustainable_harvest"] = (
                round(sustainable_total / n_agents, 2) if n_agents > 0 else 0
            )
            extra["harvest_cap"] = round(resource_before * model.harvest_cap, 2)
            extra["overusing"] = (extra["last_harvest"] or 0) > extra[
                "sustainable_harvest"
            ]
        elif game_id == "minority_game":
            last_choice = getattr(agent, "last_choice", None)
            if last_choice is True:
                extra["action"] = "A"
            elif last_choice is False:
                extra["action"] = "B"
            else:
                extra["action"] = None

        # Extract LLM error if the brain had issues this round
        brain = getattr(agent, "brain", None)
        if _is_llm_brain(brain) and brain.last_error:
            extra["llm_error"] = brain.last_error

        # Extract LLM rationale if available
        if _is_llm_brain(brain) and brain.last_response_text:
            try:
                parsed = json.loads(brain.last_response_text)
                if "decisions" in parsed:
                    extra["llm_rationale"] = [
                        d.get("rationale", "") for d in parsed["decisions"]
                    ]
                elif "rationale" in parsed:
                    extra["llm_rationale"] = parsed["rationale"]
            except (json.JSONDecodeError, TypeError):
                extra["llm_rationale"] = brain.last_response_text

        # SIR uses happiness (0-100) instead of payoff
        if game_id == "sir":
            cum = getattr(agent, "happiness", 80.0)
            rnd = getattr(agent, "happiness_change", 0.0)
        else:
            cum = getattr(agent, "cumulative_payoff", 0.0)
            rnd = getattr(agent, "round_payoff", 0.0)

        agents.append(
            {
                "id": agent.unique_id,
                "label": getattr(agent, "label", str(agent.unique_id)),
                "brain_name": getattr(agent, "brain_name", "unknown"),
                "cumulative_payoff": cum,
                "round_payoff": rnd,
                "extra": extra,
            }
        )
    return agents


def extract_model_metrics(model: Any, game_id: str) -> dict[str, float]:
    df = model.datacollector.get_model_vars_dataframe()
    if len(df) == 0:
        return {}
    last = df.iloc[-1]
    return {col: float(last[col]) for col in df.columns}


def extract_game_data(model: Any, game_id: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if game_id == "el_farol":
        data["attendance_history"] = getattr(model, "attendance_history", [])
        data["threshold"] = getattr(model, "threshold", 0)
    elif game_id == "public_goods":
        data["group_avg_history"] = getattr(model, "group_avg_history", [])
    elif game_id == "schelling":
        data["width"] = getattr(model, "width", 0)
        data["height"] = getattr(model, "height", 0)
        data["tolerance"] = getattr(model, "tolerance", 0)
        agents = list(model.agents)
        type_counts: dict[int, int] = {}
        for a in agents:
            type_counts[a.agent_type] = type_counts.get(a.agent_type, 0) + 1
        data["type_counts"] = type_counts
        data["type_a_count"] = type_counts.get(0, 0)
        data["type_b_count"] = type_counts.get(1, 0)
        data["happy_count"] = sum(1 for a in agents if a.happy)
        data["moved_count"] = sum(1 for a in agents if a.moved)
        type_stats: dict[int, dict[str, Any]] = {}
        for t, cnt in type_counts.items():
            type_agents = [a for a in agents if a.agent_type == t]
            type_stats[t] = {
                "count": cnt,
                "happy_count": sum(1 for a in type_agents if a.happy),
                "happy_rate": sum(1 for a in type_agents if a.happy) / cnt
                if cnt
                else 0,
                "moved_count": sum(1 for a in type_agents if a.moved),
                "move_rate": sum(1 for a in type_agents if a.moved) / cnt if cnt else 0,
            }
        data["type_stats"] = {str(k): v for k, v in type_stats.items()}
        n_islands, largest, avg = model._metric_islands()
        data["n_islands"] = n_islands
        data["largest_island"] = largest
        data["avg_island_size"] = avg
    elif game_id == "sir":
        data["peak_infection"] = getattr(model, "_peak_infection", 0)
        data["beta"] = getattr(model, "beta", 0)
        data["gamma"] = getattr(model, "gamma", 0)
        agents = list(model.agents)
        data["susceptible_count"] = sum(
            1 for a in agents if a.health_state == HealthState.SUSCEPTIBLE
        )
        data["infected_count"] = sum(
            1 for a in agents if a.health_state == HealthState.INFECTED
        )
        data["recovered_count"] = sum(
            1 for a in agents if a.health_state == HealthState.RECOVERED
        )
        data["isolating_count"] = sum(1 for a in agents if a.isolated)
        data["vaccinated_count"] = sum(
            1 for a in agents if getattr(a, "vaccinated", False)
        )
        data["total_agents"] = len(agents)
        import networkx as nx

        network = getattr(model, "network", None)
        if network is not None:
            degrees = [d for _, d in network.degree()]
            data["avg_degree"] = sum(degrees) / len(degrees) if degrees else 0
            data["max_degree"] = max(degrees) if degrees else 0
            data["min_degree"] = min(degrees) if degrees else 0
            data["num_edges"] = network.number_of_edges()
            data["clustering_coeff"] = nx.average_clustering(network)
            if nx.is_connected(network):
                data["avg_path_length"] = nx.average_shortest_path_length(network)
            else:
                largest_cc = max(nx.connected_components(network), key=len)
                subg = network.subgraph(largest_cc)
                data["avg_path_length"] = (
                    nx.average_shortest_path_length(subg) if len(largest_cc) > 1 else 0
                )
            data["density"] = nx.density(network)
        positions = getattr(model, "_node_positions", {})
        agent_map = getattr(model, "_agent_map", {})
        data["nodes"] = [
            {
                "id": agent.unique_id,
                "x": float(positions[node_id][0]),
                "y": float(positions[node_id][1]),
            }
            for node_id, agent in agent_map.items()
            if node_id in positions
        ]
        data["edges"] = [
            {"source": agent_map[u].unique_id, "target": agent_map[v].unique_id}
            for u, v in model.network.edges()
            if u in agent_map and v in agent_map
        ]
    return data
