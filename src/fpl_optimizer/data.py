"""Fetching and preparing data from the Fantasy Premier League API."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from . import config
from .model import shrink_per90
from .squad import norm_name

Fixture = dict  # {'opp_id': int, 'opp_name': str, 'is_home': bool}


@dataclass
class FPLData:
    players: pd.DataFrame
    target_gws: list[int]
    team_fixtures: dict[int, dict[int, list[Fixture]]]  # team_id -> gameweek -> fixtures
    team_stats: dict[int, dict[str, float]]  # team_id -> {'xg_pg', 'xga_pg'}
    league_avg_xg: float
    current_gw: int | None


def fetch(path: str) -> dict:
    resp = requests.get(config.BASE_URL + path, headers=config.REQUEST_HEADERS,
                        timeout=config.REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def load_fpl_data(horizon: int = config.DEFAULT_HORIZON) -> FPLData:
    return build_fpl_data(fetch("bootstrap-static/"), fetch("fixtures/"), horizon)


def fetch_team_squad(team_id: int, gameweek: int) -> tuple[list[int], float]:
    """Returns (player ids, money in the bank) of an FPL team."""
    data = fetch(f"entry/{team_id}/event/{gameweek}/picks/")
    ids = [pick["element"] for pick in data["picks"]]
    return ids, data["entry_history"]["bank"] / 10.0


def build_fpl_data(bootstrap: dict, fixtures: list[dict], horizon: int) -> FPLData:
    events = bootstrap["events"]
    current_gw = next((e["id"] for e in events if e.get("is_current")), None)
    next_gw = next((e["id"] for e in events if e.get("is_next")), None)
    if next_gw is None:
        next_gw = (current_gw or 0) + 1
    target_gws = [gw for gw in range(next_gw, next_gw + horizon) if gw <= 38]
    gws_played = sum(1 for e in events if e.get("finished"))

    positions = {t["id"]: t["singular_name_short"] for t in bootstrap["element_types"]}
    teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    elements = bootstrap["elements"]

    team_stats, league_avg_xg = build_team_stats(elements, fixtures, teams)
    return FPLData(
        players=build_players(elements, positions, teams, gws_played),
        target_gws=target_gws,
        team_fixtures=build_fixture_map(fixtures, target_gws, teams),
        team_stats=team_stats,
        league_avg_xg=league_avg_xg,
        current_gw=current_gw,
    )


def build_team_stats(elements, fixtures, teams) -> tuple[dict, float]:
    """Team attack (xG per game) and defence (xGA per 90 of the most-used player)."""
    matches_played = Counter()
    for f in fixtures:
        if f.get("finished"):
            matches_played[f["team_h"]] += 1
            matches_played[f["team_a"]] += 1

    team_xg = Counter()
    anchor = {}  # team_id -> (minutes, xGC) of the player with most minutes
    for p in elements:
        t_id = p["team"]
        mins = float(p.get("minutes") or 0)
        team_xg[t_id] += float(p.get("expected_goals") or 0)
        if mins > 200 and mins > anchor.get(t_id, (0.0, 0.0))[0]:
            anchor[t_id] = (mins, float(p.get("expected_goals_conceded") or 0))

    played_xg = [team_xg[t] / matches_played[t] for t in teams if matches_played[t] > 0]
    league_avg_xg = float(np.mean(played_xg)) if played_xg else 1.35

    stats = {}
    for t_id in teams:
        xg_pg = team_xg[t_id] / matches_played[t_id] if matches_played[t_id] else league_avg_xg
        if t_id in anchor:
            mins, xgc = anchor[t_id]
            xga_pg = xgc / (mins / 90.0)
        else:
            xga_pg = league_avg_xg
        stats[t_id] = {"xg_pg": xg_pg, "xga_pg": xga_pg}
    return stats, league_avg_xg


def build_fixture_map(fixtures, target_gws, teams) -> dict[int, dict[int, list[Fixture]]]:
    """Fixtures per team and gameweek; a list per gameweek so double gameweeks are kept."""
    result = {t_id: {} for t_id in teams}
    for f in fixtures:
        gw = f.get("event")
        if gw in target_gws:
            home, away = f["team_h"], f["team_a"]
            result[home].setdefault(gw, []).append(
                {"opp_id": away, "opp_name": teams[away], "is_home": True})
            result[away].setdefault(gw, []).append(
                {"opp_id": home, "opp_name": teams[home], "is_home": False})
    return result


def build_players(elements, positions, teams, gws_played) -> pd.DataFrame:
    rows = []
    for p in elements:
        pos = positions[p["element_type"]]
        cost = float(p["now_cost"]) / 10.0
        minutes = float(p.get("minutes") or 0)
        starts = float(p.get("starts") or 0)
        chance = p.get("chance_of_playing_next_round")
        chance_factor = float(chance) / 100.0 if chance is not None else 1.0

        if gws_played > 0:
            p_start = min(1.0, starts / gws_played) * chance_factor
            mins_per_start = min(90.0, minutes / starts) if starts > 0 else 0.0
            p_sub = config.SUB_APPEAR_PROB * chance_factor if minutes > 0 else 0.0
        else:
            # Pre-season: guess playing time from price
            guess = 0.0 if cost <= 4.0 else min(90.0, (cost - 4.0) * 20.0 + 20.0)
            p_start = guess / 90.0 * chance_factor
            mins_per_start = 85.0
            p_sub = 0.0

        rows.append({
            "id": p["id"],
            "web_name": p["web_name"],
            "norm_name": norm_name(p["web_name"]),
            "position": pos,
            "team_id": p["team"],
            "team_name": teams.get(p["team"], "UNK"),
            "cost": cost,
            "p_start": p_start,
            "p_sub": p_sub,
            "mins_per_start": mins_per_start,
            "x_mins": p_start * mins_per_start + (1 - p_start) * p_sub * config.SUB_MINUTES,
            "xg_90": shrink_per90(float(p.get("expected_goals") or 0), minutes,
                                  config.POS_BASELINE_XG[pos]),
            "xa_90": shrink_per90(float(p.get("expected_assists") or 0), minutes,
                                  config.POS_BASELINE_XA[pos]),
            "dc_90": shrink_per90(float(p.get("defensive_contribution") or 0), minutes,
                                  config.POS_BASELINE_DC[pos]),
        })
    return pd.DataFrame(rows)
