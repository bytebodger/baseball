"""Resolves a game's park_id from its home_team code and season.

Statcast's raw pull has no venue/park field at all, but a team's home park
almost never changes -- the only relocations affecting this project's raw
data (2010-2026 pulled, 2015-2026 actually used for modeling) are the
Marlins (2012), Braves (2017), Rangers (2020), and Athletics (2025), all
captured in configs/park_history.yaml. Every other team's park_id is just
its own team code: one park for the whole range needs no special-casing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "park_history.yaml"


def load_park_history(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, list[dict]]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["park_history"]


def _park_id_for(team: str, season: int, park_history: dict[str, list[dict]]) -> str:
    entries = park_history.get(team)
    if entries is None:
        return team

    applicable = [e for e in entries if e["from_season"] <= season]
    if not applicable:
        # A season earlier than this team's earliest listed entry: assume
        # that earliest-known park was also its park going further back --
        # true for every team in this crosswalk today (each one's first
        # entry is that franchise's park from well before our raw data
        # starts), not just "since our data happens to start there." If a
        # team ever needs an even-earlier park distinguished, add an entry
        # for it rather than relying on this fallback.
        return min(entries, key=lambda e: e["from_season"])["park_id"]
    return max(applicable, key=lambda e: e["from_season"])["park_id"]


def resolve_park_id(home_team: pd.Series, season: pd.Series, park_history: dict[str, list[dict]]) -> pd.Series:
    """Vectorized via a small (team, season) -> park_id lookup table built
    from just the unique combinations actually present, rather than calling
    _park_id_for once per row -- there are at most 30 teams x N seasons of
    unique combinations, orders of magnitude fewer than the row count."""
    pairs = pd.DataFrame({"team": home_team.to_numpy(), "season": season.to_numpy()})
    unique_pairs = pairs.drop_duplicates().reset_index(drop=True)
    unique_pairs["park_id"] = [
        _park_id_for(team, int(season_), park_history)
        for team, season_ in zip(unique_pairs["team"], unique_pairs["season"])
    ]
    merged = pairs.merge(unique_pairs, on=["team", "season"], how="left")
    return pd.Series(merged["park_id"].to_numpy(), index=home_team.index, name="park_id")
