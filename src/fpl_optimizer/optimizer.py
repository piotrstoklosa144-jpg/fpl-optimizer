"""Mixed-integer linear programming (MILP) models for transfers and line-up selection."""

from __future__ import annotations

import pandas as pd
import pulp
from pulp import LpMaximize, LpProblem, lpSum

from . import config

# Compatible with PuLP 2.x-4.x: HiGHS solver (via `highspy`) when available, else bundled CBC.
_HIGHS = pulp.HiGHS(msg=False)
SOLVER = _HIGHS if _HIGHS.available() else pulp.PULP_CBC_CMD(msg=False)
_OPTIMAL = 1  # same status code in every PuLP version


def _binary_var(prob: LpProblem, name: str):
    if hasattr(prob, "add_variable"):  # PuLP >= 3.3
        return prob.add_variable(name, cat="Binary")
    return pulp.LpVariable(name, cat="Binary")


def _solve(prob: LpProblem) -> bool:
    """Solves the model; returns True if an optimal solution was found."""
    result = prob.solve(SOLVER)
    status = getattr(result, "status", result)  # PuLP 4 returns a stats object
    return getattr(status, "value", status) == _OPTIMAL


def _add_formation_constraints(prob: LpProblem, start_vars: dict, positions: pd.Series) -> None:
    """Starting XI: 1 GKP, 3-5 DEF, 3-5 MID, 1-3 FWD."""
    prob += lpSum(start_vars.values()) == 11
    for pos, (lo, hi) in config.FORMATION_LIMITS.items():
        n_starting = lpSum(v for i, v in start_vars.items() if positions[i] == pos)
        prob += n_starting >= lo
        prob += n_starting <= hi


def optimize_transfers(
    players: pd.DataFrame,
    squad_ids: list[int],
    budget: float,
    max_transfers: int = 1,
    metric: str = "xp_total",
    bench_weight: float = 0.05,
    max_expensive_defs: int | None = 2,
    expensive_def_price: float = 5.5,
) -> pd.DataFrame:
    """Returns the best 15-player squad reachable with at most `max_transfers` transfers."""
    # Players without expected points can't improve the squad - smaller model, same optimum.
    cand = players[(players[metric] > 0) | players["id"].isin(squad_ids)]
    idx = list(cand.index)
    xp, cost, pos, team, pid = cand[metric], cand["cost"], cand["position"], cand["team_id"], cand["id"]
    current = set(squad_ids)

    prob = LpProblem("fpl_transfers", LpMaximize)
    squad = {i: _binary_var(prob, f"squad_{i}") for i in idx}
    start = {i: _binary_var(prob, f"start_{i}") for i in idx}
    capt = {i: _binary_var(prob, f"capt_{i}") for i in idx}

    # Objective: starting XI + captain bonus + a small weight for bench depth
    prob += (
        lpSum(start[i] * xp[i] for i in idx)
        + lpSum(capt[i] * xp[i] for i in idx)
        + bench_weight * lpSum((squad[i] - start[i]) * xp[i] for i in idx)
    )

    for i in idx:
        prob += start[i] <= squad[i]
        prob += capt[i] <= start[i]

    prob += lpSum(squad[i] for i in idx if pid[i] in current) >= config.SQUAD_SIZE - max_transfers
    prob += lpSum(squad[i] * cost[i] for i in idx) <= budget
    prob += lpSum(squad.values()) == config.SQUAD_SIZE
    for p, n in config.SQUAD_SLOTS.items():
        prob += lpSum(squad[i] for i in idx if pos[i] == p) == n
    for t_id in team.unique():
        prob += lpSum(squad[i] for i in idx if team[i] == t_id) <= config.MAX_PER_TEAM
    if max_expensive_defs is not None:
        prob += lpSum(
            squad[i] for i in idx if pos[i] == "DEF" and cost[i] > expensive_def_price
        ) <= max_expensive_defs

    _add_formation_constraints(prob, start, pos)
    prob += lpSum(capt.values()) == 1

    if not _solve(prob):
        raise ValueError("No optimal solution found for the transfer problem.")
    return players.loc[[i for i in idx if squad[i].varValue > 0.5]].copy()


def pick_lineup(squad: pd.DataFrame, metric: str = "xp_total") -> pd.DataFrame:
    """Chooses the starting XI, captain and vice-captain; returns the squad ordered for display."""
    idx = squad.index
    xp, pos = squad[metric], squad["position"]

    prob = LpProblem("fpl_lineup", LpMaximize)
    start = {i: _binary_var(prob, f"start_{i}") for i in idx}
    capt = {i: _binary_var(prob, f"capt_{i}") for i in idx}
    vice = {i: _binary_var(prob, f"vice_{i}") for i in idx}

    prob += lpSum(start[i] * xp[i] + capt[i] * xp[i] + 0.01 * vice[i] * xp[i] for i in idx)
    for i in idx:
        prob += capt[i] <= start[i]
        prob += vice[i] <= start[i]
        prob += capt[i] + vice[i] <= 1
    _add_formation_constraints(prob, start, pos)
    prob += lpSum(capt.values()) == 1
    prob += lpSum(vice.values()) == 1

    if not _solve(prob):
        raise ValueError("No valid line-up found for this squad.")

    res = squad.copy()
    res["starting"] = [round(start[i].varValue) for i in idx]
    res["captain"] = [round(capt[i].varValue) for i in idx]
    res["vice_captain"] = [round(vice[i].varValue) for i in idx]
    res["role"] = "Starter"
    res.loc[res["captain"] == 1, "role"] = "Captain (x2)"
    res.loc[res["vice_captain"] == 1, "role"] = "Vice-captain"

    res["_pos_order"] = res["position"].map(config.POS_ORDER)
    starters = res[res["starting"] == 1].sort_values(["_pos_order", metric], ascending=[True, False])
    bench_gkp = res[(res["starting"] == 0) & (res["position"] == "GKP")].copy()
    bench_out = res[(res["starting"] == 0) & (res["position"] != "GKP")].sort_values(metric, ascending=False)
    bench_gkp["role"] = "Bench (GKP)"
    bench_out["role"] = [f"Bench #{n}" for n in range(1, len(bench_out) + 1)]
    return pd.concat([starters, bench_gkp, bench_out]).drop(columns="_pos_order")


def lineup_points(lineup: pd.DataFrame, metric: str = "xp_total") -> float:
    """Expected points of the starting XI with the captain counted twice."""
    return float(
        lineup.loc[lineup["starting"] == 1, metric].sum() + lineup.loc[lineup["captain"] == 1, metric].sum()
    )
