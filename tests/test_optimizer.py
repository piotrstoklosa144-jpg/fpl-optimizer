from fpl_optimizer import config
from fpl_optimizer.optimizer import lineup_points, optimize_transfers, pick_lineup


def test_finds_the_obvious_upgrade(pool, budget, squad_ids):
    new = optimize_transfers(pool, squad_ids, budget, max_transfers=1)
    assert 20 in set(new["id"])
    assert 12 not in set(new["id"])


def test_respects_squad_rules(pool, budget, squad_ids):
    new = optimize_transfers(pool, squad_ids, budget, max_transfers=1)
    assert len(new) == config.SQUAD_SIZE
    assert new["position"].value_counts().to_dict() == config.SQUAD_SLOTS
    assert new["cost"].sum() <= budget
    assert new["team_id"].value_counts().max() <= config.MAX_PER_TEAM
    assert len(set(new["id"]) - set(squad_ids)) <= 1


def test_does_not_break_three_per_team_limit(pool, budget, squad_ids):
    # Player 22 is the best keeper but would be a 4th player from team 3
    new = optimize_transfers(pool, squad_ids, budget, max_transfers=1)
    assert 22 not in set(new["id"])


def test_zero_transfers_keeps_the_squad(pool, budget, squad_ids):
    new = optimize_transfers(pool, squad_ids, budget, max_transfers=0)
    assert set(new["id"]) == set(squad_ids)


def test_lineup_is_a_valid_formation(pool, squad_ids):
    lineup = pick_lineup(pool[pool["id"].isin(squad_ids)])
    starters = lineup[lineup["starting"] == 1]
    counts = starters["position"].value_counts().to_dict()

    assert len(starters) == 11
    for pos, (lo, hi) in config.FORMATION_LIMITS.items():
        assert lo <= counts.get(pos, 0) <= hi
    assert lineup["captain"].sum() == 1
    assert lineup["vice_captain"].sum() == 1


def test_captain_is_the_best_starter(pool, squad_ids):
    lineup = pick_lineup(pool[pool["id"].isin(squad_ids)])
    captain = lineup[lineup["captain"] == 1].iloc[0]
    assert captain["xp_total"] == lineup.loc[lineup["starting"] == 1, "xp_total"].max()


def test_lineup_points_counts_captain_twice(pool, squad_ids):
    lineup = pick_lineup(pool[pool["id"].isin(squad_ids)])
    starters_xp = lineup.loc[lineup["starting"] == 1, "xp_total"].sum()
    captain_xp = lineup.loc[lineup["captain"] == 1, "xp_total"].iloc[0]
    assert lineup_points(lineup) == starters_xp + captain_xp
