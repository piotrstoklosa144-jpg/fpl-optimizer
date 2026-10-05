"""Synthetic data so tests run offline and are deterministic."""

import pandas as pd
import pytest

from fpl_optimizer.data import FPLData
from fpl_optimizer.squad import norm_name


def make_player(pid, pos, team_id, cost, xp, name=None, **overrides):
    player = {
        "id": pid,
        "web_name": name or f"P{pid}",
        "norm_name": (name or f"P{pid}").lower(),
        "position": pos,
        "team_id": team_id,
        "team_name": f"T{team_id}",
        "cost": cost,
        "p_start": 0.9,
        "p_sub": 0.25,
        "mins_per_start": 85.0,
        "x_mins": 80.0,
        "xg_90": 0.2,
        "xa_90": 0.1,
        "dc_90": 8.0,
        "xp_total": xp,
    }
    player.update(overrides)
    return player


# 15-player squad spread over 6 teams (max 3 per team)
SQUAD = [
    make_player(1, "GKP", 1, 5.0, 20), make_player(2, "GKP", 2, 4.0, 2),
    make_player(3, "DEF", 1, 5.0, 18), make_player(4, "DEF", 2, 4.5, 16),
    make_player(5, "DEF", 3, 4.5, 15), make_player(6, "DEF", 4, 4.0, 12),
    make_player(7, "DEF", 5, 4.0, 5),
    make_player(8, "MID", 3, 8.0, 30), make_player(9, "MID", 4, 7.0, 25),
    make_player(10, "MID", 5, 6.5, 22), make_player(11, "MID", 6, 6.0, 20),
    make_player(12, "MID", 6, 5.5, 10),
    make_player(13, "FWD", 1, 9.0, 28), make_player(14, "FWD", 2, 7.5, 24),
    make_player(15, "FWD", 3, 6.0, 19),
]
SQUAD_IDS = [p["id"] for p in SQUAD]

MARKET = [
    # Clear upgrade on the weakest midfielder (id 12), affordable
    make_player(20, "MID", 7, 6.0, 40),
    # Even better but far too expensive
    make_player(21, "FWD", 7, 25.0, 100),
    # Great keeper, but team 3 already has 3 players in the squad
    make_player(22, "GKP", 3, 4.0, 50),
]


@pytest.fixture
def squad_ids():
    return list(SQUAD_IDS)


@pytest.fixture
def squad():
    return pd.DataFrame(SQUAD)


@pytest.fixture
def named_players():
    """Two players called Thomas and one name with an accent."""
    return pd.DataFrame([
        make_player(1, "DEF", 1, 4.0, 0, name="Thomas"),
        make_player(2, "DEF", 2, 4.0, 0, name="Thomas"),
        make_player(3, "DEF", 3, 6.0, 0, name="Guéhi", norm_name=norm_name("Guéhi")),
    ])


@pytest.fixture
def pool():
    return pd.DataFrame(SQUAD + MARKET)


@pytest.fixture
def budget():
    return sum(p["cost"] for p in SQUAD) + 0.5


@pytest.fixture
def fpl_data():
    """Two teams; team 1 has a double gameweek in GW 2."""
    players = pd.DataFrame([
        make_player(1, "DEF", 1, 5.0, 0, p_start=0.8, xg_90=0.1, xa_90=0.1, dc_90=9.0),
        make_player(2, "DEF", 1, 5.0, 0, p_start=1.0),
        make_player(3, "FWD", 2, 8.0, 0, p_start=0.95, xg_90=0.6, xa_90=0.2),
    ])
    return FPLData(
        players=players,
        target_gws=[1, 2],
        team_fixtures={
            1: {1: [{"opp_id": 2, "opp_name": "T2", "is_home": True}],
                2: [{"opp_id": 2, "opp_name": "T2", "is_home": False},
                    {"opp_id": 2, "opp_name": "T2", "is_home": True}]},
            2: {1: [{"opp_id": 1, "opp_name": "T1", "is_home": False}],
                2: [{"opp_id": 1, "opp_name": "T1", "is_home": True}]},
        },
        team_stats={1: {"xg_pg": 1.2, "xga_pg": 1.0}, 2: {"xg_pg": 1.8, "xga_pg": 1.5}},
        league_avg_xg=1.4,
        current_gw=0,
    )
