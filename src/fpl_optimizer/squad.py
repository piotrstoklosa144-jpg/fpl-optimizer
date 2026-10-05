"""Resolving and validating the user's 15-player squad."""

from __future__ import annotations

import unicodedata

import pandas as pd

from . import config


def norm_name(name: str) -> str:
    """Lower-case name without accents, so 'Guehi' matches 'Guéhi'."""
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().strip()


def resolve_squad_names(players: pd.DataFrame, entries: list[str]) -> list[int]:
    """Maps entries like 'Haaland' or 'Thomas|COV' to player ids; fails on ambiguous names."""
    ids = []
    for entry in entries:
        name, _, team = entry.partition("|")
        mask = players["norm_name"] == norm_name(name)
        if team:
            mask &= players["team_name"] == team.strip().upper()
        hits = players[mask]
        if len(hits) != 1:
            similar = players[players["norm_name"].str.contains(norm_name(name), regex=False)]
            options = ", ".join(f"{r.web_name}|{r.team_name}" for r in similar.itertuples()) or "none"
            raise ValueError(f"'{entry}' matches {len(hits)} players. Similar: {options}")
        ids.append(int(hits["id"].iloc[0]))
    return ids


def validate_squad(players: pd.DataFrame, ids: list[int]) -> None:
    squad = players[players["id"].isin(ids)]
    counts = squad["position"].value_counts().to_dict()
    if len(set(ids)) != config.SQUAD_SIZE or any(
        counts.get(pos, 0) != n for pos, n in config.SQUAD_SLOTS.items()
    ):
        raise ValueError(f"Squad must have 15 players (2 GKP / 5 DEF / 5 MID / 3 FWD), got: {counts}")


def read_squad_file(path) -> list[str]:
    """One player per line; blank lines and lines starting with '#' are ignored."""
    with open(path, encoding="utf-8") as f:
        lines = (line.strip() for line in f)
        return [line for line in lines if line and not line.startswith("#")]
