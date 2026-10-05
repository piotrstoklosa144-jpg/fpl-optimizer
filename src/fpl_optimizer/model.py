"""Expected-points model: goals, assists and goals conceded follow Poisson distributions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    import pandas as pd

    from .data import FPLData


def shrink_per90(total: float, minutes: float, baseline: float) -> float:
    """Bayesian shrinkage: with few minutes the rate stays close to the positional baseline."""
    prior_90s = config.PRIOR_MINS / 90.0
    return (total + prior_90s * baseline) / (minutes / 90.0 + prior_90s)


def poisson_at_least(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    cdf = sum(math.exp(-lam) * lam**i / math.factorial(i) for i in range(k))
    return max(0.0, 1.0 - cdf)


def expected_half_floor(lam: float) -> float:
    """E[floor(G / 2)] for G ~ Poisson(lam): FPL deducts 1 point per 2 goals conceded."""
    p_odd = (1 - math.exp(-2 * lam)) / 2
    return (lam - p_odd) / 2


def match_params(player, fixture, team_stats, league_avg_xg) -> dict:
    """Per-match parameters shared by the analytical model and the Monte Carlo simulation."""
    opp = team_stats[fixture["opp_id"]]
    own = team_stats[player["team_id"]]
    venue = config.HOME_ADV if fixture["is_home"] else config.AWAY_ADV

    att_mult = opp["xga_pg"] / league_avg_xg * venue
    start_frac = player["mins_per_start"] / 90.0
    pos = player["position"]
    p_dc = 0.0
    if pos != "GKP":
        p_dc = poisson_at_least(config.DC_THRESHOLD[pos], player["dc_90"] * start_frac)

    return {
        "p_start": player["p_start"],
        "p_sub": (1 - player["p_start"]) * player["p_sub"],
        "start_frac": start_frac,
        "sub_frac": config.SUB_MINUTES / 90.0,
        "lam_g90": player["xg_90"] * att_mult,
        "lam_a90": player["xa_90"] * att_mult,
        # expected goals conceded by the player's team over the whole match
        "lam_conc": opp["xg_pg"] * own["xga_pg"] / league_avg_xg / venue,
        "p_dc": p_dc,
    }


def expected_points(pos: str, p: dict) -> float:
    """Expected FPL points in one match."""
    s, b = p["p_start"], p["p_sub"]
    if s + b <= 0:
        return 0.0
    frac = s * p["start_frac"] + b * p["sub_frac"]
    e_goals = p["lam_g90"] * frac
    e_assists = p["lam_a90"] * frac
    lam = p["lam_conc"]
    p_cs = s * math.exp(-lam)  # Poisson P(0 conceded); clean sheets only for starters

    pts = 2.0 * s + 1.0 * b
    pts += e_goals * config.GOAL_PTS[pos] + e_assists * config.ASSIST_PTS + p_cs * config.CS_PTS[pos]
    if pos in ("GKP", "DEF"):
        pts -= s * expected_half_floor(lam)
    pts += s * p["p_dc"] * config.DC_PTS
    pts -= config.CARD_PENALTY_90 * frac
    pts += min(config.MAX_BONUS, 1.2 * e_goals + 0.8 * e_assists + 0.5 * p_cs)
    return pts


def add_expected_points(data: FPLData) -> pd.DataFrame:
    """Adds xp_next (next gameweek), xp_total (whole horizon) and next_opponent columns."""
    df = data.players.copy()
    xp_next, xp_total, next_opp = [], [], []

    for _, player in df.iterrows():
        total, first_xp, first_opp = 0.0, 0.0, "-"
        for idx, gw in enumerate(data.target_gws):
            fixtures = data.team_fixtures[player["team_id"]].get(gw, [])
            gw_xp = sum(
                expected_points(player["position"],
                                match_params(player, fix, data.team_stats, data.league_avg_xg))
                for fix in fixtures
            )
            total += gw_xp
            if idx == 0 and fixtures:
                first_xp = gw_xp
                first_opp = ", ".join(f"{f['opp_name']} ({'H' if f['is_home'] else 'A'})" for f in fixtures)
        xp_next.append(first_xp)
        xp_total.append(total)
        next_opp.append(first_opp)

    df["xp_next"] = xp_next
    df["xp_total"] = xp_total
    df["next_opponent"] = next_opp
    return df
