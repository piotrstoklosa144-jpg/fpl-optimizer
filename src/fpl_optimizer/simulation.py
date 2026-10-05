"""Monte Carlo simulation of FPL points to quantify risk."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import config
from .model import match_params

if TYPE_CHECKING:
    from .data import FPLData


def simulate_points(
    players: pd.DataFrame,
    player_ids: Iterable[int],
    data: FPLData,
    n_sims: int = config.DEFAULT_N_SIMS,
    seed: int = config.DEFAULT_SEED,
) -> dict[int, np.ndarray]:
    """Samples each player's total points over the horizon -> {id: array(n_sims)}.

    Goals conceded are drawn once per team and match, so clean sheets of
    players from the same team are correlated, as in reality.
    """
    rng = np.random.default_rng(seed)
    by_id = players.set_index("id")
    conceded = {}
    sims = {}

    for pid in player_ids:
        player = by_id.loc[pid]
        pos, team_id = player["position"], player["team_id"]
        total = np.zeros(n_sims)

        for gw in data.target_gws:
            for k, fix in enumerate(data.team_fixtures[team_id].get(gw, [])):
                p = match_params(player, fix, data.team_stats, data.league_avg_xg)
                key = (team_id, gw, k)
                if key not in conceded:
                    conceded[key] = rng.poisson(p["lam_conc"], n_sims)
                total += _simulate_match(rng, pos, p, conceded[key], n_sims)

        sims[pid] = total
    return sims


def _simulate_match(rng, pos: str, p: dict, goals_conceded: np.ndarray, n_sims: int) -> np.ndarray:
    u = rng.random(n_sims)
    started = u < p["p_start"]
    subbed = ~started & (u < p["p_start"] + p["p_sub"])
    frac = np.where(started, p["start_frac"], np.where(subbed, p["sub_frac"], 0.0))

    goals = rng.poisson(p["lam_g90"] * frac)
    assists = rng.poisson(p["lam_a90"] * frac)
    clean_sheet = started & (goals_conceded == 0)

    pts = 2.0 * started + 1.0 * subbed
    pts += goals * config.GOAL_PTS[pos] + assists * config.ASSIST_PTS + clean_sheet * config.CS_PTS[pos]
    if pos in ("GKP", "DEF"):
        pts -= started * (goals_conceded // 2)
    pts += config.DC_PTS * (started & (rng.random(n_sims) < p["p_dc"]))
    pts -= config.CARD_PENALTY_90 * frac
    pts += np.minimum(config.MAX_BONUS, 1.2 * goals + 0.8 * assists + 0.5 * clean_sheet)
    return pts


def lineup_distribution(lineup: pd.DataFrame, sims: dict[int, np.ndarray]) -> np.ndarray:
    """Points of the starting XI plus the captain (counted twice) in every simulation."""
    starters = lineup.loc[lineup["starting"] == 1, "id"]
    captain = lineup.loc[lineup["captain"] == 1, "id"].iloc[0]
    return sum(sims[pid] for pid in starters) + sims[captain]


def summarize(dist: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(dist.mean()),
        "median": float(np.median(dist)),
        "p10": float(np.percentile(dist, 10)),
        "p90": float(np.percentile(dist, 90)),
        "std": float(dist.std()),
    }
