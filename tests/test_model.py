import math

import numpy as np
import pytest

from fpl_optimizer import config
from fpl_optimizer.model import (
    add_expected_points,
    expected_half_floor,
    expected_points,
    match_params,
    poisson_at_least,
    shrink_per90,
)
from fpl_optimizer.simulation import simulate_points


def test_shrinkage_without_minutes_returns_baseline():
    assert shrink_per90(0, 0, 0.25) == pytest.approx(0.25)


def test_shrinkage_converges_to_observed_rate():
    # 50 goals in 100 full matches -> close to 0.5 per 90 despite a 0.1 prior
    assert shrink_per90(50, 9000, 0.1) == pytest.approx(0.5, abs=0.02)


def test_poisson_at_least():
    assert poisson_at_least(0, 2.0) == 1.0
    assert poisson_at_least(1, 2.0) == pytest.approx(1 - math.exp(-2.0))
    assert poisson_at_least(3, 0.0) == 0.0


@pytest.mark.parametrize("lam", [0.3, 1.0, 2.5])
def test_expected_half_floor_matches_sampling(lam):
    samples = np.random.default_rng(0).poisson(lam, 200_000)
    assert expected_half_floor(lam) == pytest.approx((samples // 2).mean(), abs=0.01)


def test_player_who_does_not_play_scores_zero():
    params = {"p_start": 0.0, "p_sub": 0.0}
    assert expected_points("MID", params) == 0.0


def test_easier_opponent_gives_more_points(fpl_data):
    striker = fpl_data.players.iloc[2]
    weak = {**fpl_data.team_stats, 1: {"xg_pg": 1.0, "xga_pg": 2.0}}
    strong = {**fpl_data.team_stats, 1: {"xg_pg": 1.0, "xga_pg": 0.7}}
    fixture = {"opp_id": 1, "is_home": True}
    xp_weak = expected_points("FWD", match_params(striker, fixture, weak, 1.4))
    xp_strong = expected_points("FWD", match_params(striker, fixture, strong, 1.4))
    assert xp_weak > xp_strong


def test_double_gameweek_counts_both_matches(fpl_data):
    df = add_expected_points(fpl_data)
    defender, striker = df.iloc[0], df.iloc[2]
    # team 1 plays three matches in the horizon, team 2 only two
    assert defender["xp_total"] > 2.5 * defender["xp_next"]
    assert striker["xp_total"] < 2.5 * striker["xp_next"]


def test_monte_carlo_mean_matches_analytical_model(fpl_data):
    df = add_expected_points(fpl_data)
    sims = simulate_points(df, df["id"], fpl_data, n_sims=200_000, seed=1)
    for _, player in df.iterrows():
        assert sims[player["id"]].mean() == pytest.approx(player["xp_total"], rel=0.03)


def test_clean_sheets_are_correlated_within_a_team(fpl_data):
    df = add_expected_points(fpl_data)
    sims = simulate_points(df, [1, 2, 3], fpl_data, n_sims=50_000, seed=1)
    same_team = np.corrcoef(sims[1], sims[2])[0, 1]  # two defenders of team 1
    other_team = np.corrcoef(sims[1], sims[3])[0, 1]  # defender vs. striker of team 2
    assert same_team > 0.15
    assert abs(other_team) < 0.02


def test_goal_points_follow_fpl_rules():
    assert config.GOAL_PTS == {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
