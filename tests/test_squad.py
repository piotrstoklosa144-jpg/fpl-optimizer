import pytest

from fpl_optimizer.squad import read_squad_file, resolve_squad_names, validate_squad


def test_ambiguous_name_raises_with_hint(named_players):
    with pytest.raises(ValueError, match="matches 2 players"):
        resolve_squad_names(named_players, ["Thomas"])


def test_team_suffix_resolves_ambiguity(named_players):
    assert resolve_squad_names(named_players, ["Thomas|T2"]) == [2]


def test_name_matching_ignores_accents(named_players):
    assert resolve_squad_names(named_players, ["Guehi"]) == [3]


def test_validate_squad_rejects_wrong_shape(squad, squad_ids):
    validate_squad(squad, squad_ids)
    with pytest.raises(ValueError, match="15 players"):
        validate_squad(squad, squad_ids[:-1])


def test_read_squad_file_skips_comments(tmp_path):
    path = tmp_path / "squad.txt"
    path.write_text("# my team\nRaya\n\nThomas|COV\n", encoding="utf-8")
    assert read_squad_file(path) == ["Raya", "Thomas|COV"]
