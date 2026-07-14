"""Park factors: how much a park inflates or suppresses home runs and total
runs relative to the rest of the league, computed over a rolling multi-year
window and exposed as a *learned* embedding lookup keyed by (park_id,
season) -- not a single static number per park.

A static per-park number can't represent that a park's characteristics
change over time: humidor adoption (a single park in 2002, every park from
2022 -- see UNIVERSAL_HUMIDOR_SEASON in statcast_common.py), a fence or
dimension change, a relocation to a new building at a different elevation
(see park_history.py). Indexing by (park_id, season) instead of just
park_id lets each season's row drift independently, and seeding those rows
from the actual computed rate -- rather than random init -- means a
downstream model starts from a sensible prior and only has to *learn a
correction* on top of real signal, not rediscover it from scratch.

This module is intentionally standalone: it computes park factors from the
processed per-pitch table (the same one build_features.py writes) and
exposes ParkFactorEmbedding for a later phase to consume directly. It does
not wire into game_dataset.py, game_predictor.py, or anything else -- that's
Phase 4's job (the situational/event model), not this one's.

Definition used here is deliberately the simple one: a park's HR (or total
runs) rate is HRs-per-game (or runs-per-game) at that park, and its "factor"
is that rate divided by the league-wide rate over the *same* rolling window.
This is not the classic Bill James-style park factor, which compares a
team's home vs. road performance to net out team-quality effects before
attributing the rest to the park -- that adjustment is a reasonable future
improvement, but the rate-relative-to-league-average version asked for here
is simpler, still directionally correct (a true hitter's park pulls the
raw rate above league average regardless of which teams play there), and
doesn't require a home/road split of every team-season.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.data.statcast_common import PROCESSED_DATA_DIR, read_partitioned

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "park_factors.yaml"
DEFAULT_PITCHES_DIR = PROCESSED_DATA_DIR / "pitches"

DEFAULT_ROLLING_YEARS = 3
DEFAULT_EMBEDDING_DIM = 8

# Row 0 of ParkFactorEmbedding's table: reserved for a (park_id, season)
# combination we have no computed factor for at all (a park_id never seen in
# the data used to build the embedding -- e.g. a brand-new expansion team).
# Left at its small-random init rather than seeded from real stats, since
# there are none to seed it from.
UNKNOWN_PARK_INDEX = 0


# ---------------------------------------------------------------------------
# Pure pandas: raw stats computation, no torch involved.
# ---------------------------------------------------------------------------


def compute_game_totals(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per game_pk: its park, season, and that game's final tallies.

    home_score/away_score in the per-pitch table are each pitch's cumulative
    in-game score, not the final score -- but scores only ever increase
    within a game, so the max of each within a game_pk *is* that game's
    final score.
    """
    games = (
        pitches.groupby("game_pk")
        .agg(
            park_id=("park_id", "first"),
            season=("season", "first"),
            home_score=("home_score", "max"),
            away_score=("away_score", "max"),
            home_runs=("outcome", lambda s: int((s == "home_run").sum())),
        )
        .reset_index()
    )
    games["total_runs"] = games["home_score"] + games["away_score"]
    return games[["game_pk", "park_id", "season", "total_runs", "home_runs"]]


def compute_season_park_totals(game_totals: pd.DataFrame) -> pd.DataFrame:
    """One row per (park_id, season): games hosted and combined tallies."""
    return (
        game_totals.groupby(["park_id", "season"])
        .agg(
            n_games=("game_pk", "nunique"),
            total_runs=("total_runs", "sum"),
            total_home_runs=("home_runs", "sum"),
        )
        .reset_index()
        .sort_values(["park_id", "season"])
        .reset_index(drop=True)
    )


_ROLLING_TOTAL_COLS = ["n_games", "total_runs", "total_home_runs"]


def compute_league_rolling_rates(
    season_park_totals: pd.DataFrame, rolling_years: int = DEFAULT_ROLLING_YEARS
) -> pd.DataFrame:
    """One row per season: the league-wide HR rate and runs rate over the
    trailing `rolling_years` seasons *strictly before* it (summed across
    every park, not just one) -- the same denominator every park's row in
    compute_rolling_park_factors is compared against, exposed standalone so
    it can be used directly as a continuous context feature rather than only
    ever seen as a ratio's denominator.

    Strictly before, not "ending at": a game played during season S can only
    have happened after some but not all of season S's other games are in
    the books, so season S's own full-season totals are never a legitimate
    pre-game feature for a game played *in* season S -- using them would be
    look-ahead leakage the same way it would be for a player's own future
    pitches (see PlayerPitchSequenceDataset's cutoff filter). The window
    used for season S is therefore [S - rolling_years, S - 1]. This also
    means the very first season present in `season_park_totals` (and any
    season with fewer than 1 valid predecessor of its own -- doesn't apply
    league-wide unless the input starts mid-history) has no prior seasons to
    roll over and is dropped from the result rather than returned as a
    (falsely self-derived) 0-or-partial-window row.

    Deliberately reuses the *same* rolling_years window as the per-park
    factors, for consistency with them (see compute_rolling_park_factors'
    docstring: comparing a park's rate to a league rate computed over a
    different span would be comparing two different eras). Worth knowing if
    this is read directly rather than only as hr_factor's denominator: a
    single season already has ~2,430 games behind it league-wide, none of
    the small-sample-per-park motivation for smoothing over multiple years
    applies as strongly here -- reusing rolling_years still means this
    lags a true step change (e.g. a rules change effective a single season)
    by up to rolling_years-1 seasons, diluted by the pre-change seasons
    still in the window, before it's fully reflected.
    """
    league_by_season = (
        season_park_totals.groupby("season", as_index=False)[_ROLLING_TOTAL_COLS]
        .sum()
        .sort_values("season")
        .reset_index(drop=True)
    )
    # shift(1) before rolling: at row i this makes window [i-W+1, i] over the
    # *shifted* series sum original rows [i-W, i-1] -- i.e. strictly prior
    # seasons only, never row i's own season.
    prior = league_by_season[_ROLLING_TOTAL_COLS].shift(1)
    league_rolled = prior.rolling(window=rolling_years, min_periods=1).sum()
    league_by_season["league_hr_rate"] = league_rolled["total_home_runs"] / league_rolled["n_games"]
    league_by_season["league_runs_rate"] = league_rolled["total_runs"] / league_rolled["n_games"]
    # The first season has no prior season at all -- shift(1) leaves it NaN
    # (min_periods=1 requires >=1 non-null observation in the window, and a
    # single NaN doesn't count), so it's dropped rather than kept as an
    # undefined or leaked row.
    result = league_by_season.dropna(subset=["league_hr_rate", "league_runs_rate"])
    return result[["season", "league_hr_rate", "league_runs_rate"]].reset_index(drop=True)


def league_rates_for(season: pd.Series, league_rates: pd.DataFrame) -> pd.DataFrame:
    """Vectorized season -> (league_hr_rate, league_runs_rate) lookup, for
    merging this onto a game table's `season` column when assembling context
    features. A season after the latest one `league_rates` covers (a
    game whose season isn't fully aggregated yet) falls back to the latest
    known season's rates; a season before the earliest one falls back to the
    earliest -- the same both-directions fallback ParkFactorEmbedding.index_for
    uses, for the same reason.

    Returns a DataFrame with columns league_hr_rate/league_runs_rate, same
    length and index as `season`.
    """
    known = league_rates.sort_values("season").reset_index(drop=True)
    unique_seasons = pd.DataFrame({"season": season.unique()}).sort_values("season").reset_index(drop=True)
    resolved = pd.merge_asof(unique_seasons, known, on="season", direction="backward")
    if resolved["league_hr_rate"].isna().any():
        earliest = known.iloc[0]
        resolved["league_hr_rate"] = resolved["league_hr_rate"].fillna(earliest["league_hr_rate"])
        resolved["league_runs_rate"] = resolved["league_runs_rate"].fillna(earliest["league_runs_rate"])

    merged = pd.DataFrame({"season": season.to_numpy()}).merge(resolved, on="season", how="left")
    return pd.DataFrame(
        {
            "league_hr_rate": merged["league_hr_rate"].to_numpy(),
            "league_runs_rate": merged["league_runs_rate"].to_numpy(),
        },
        index=season.index,
    )


def compute_rolling_park_factors(
    season_park_totals: pd.DataFrame, rolling_years: int = DEFAULT_ROLLING_YEARS
) -> pd.DataFrame:
    """For each (park_id, season), sum that park's own trailing `rolling_years`
    seasons *strictly before* it (an expanding window for a park's first
    `rolling_years` - 1 seasons that have any predecessor at all, e.g. a
    just-relocated team, rather than requiring a full window before
    producing anything) and divide by games to get that window's HR rate and
    runs rate. Each row's "league average" is compute_league_rolling_rates'
    same strictly-prior-seasons window ending before that row's season, so
    the two rates being compared always cover an identical span of time.

    Strictly before, not "ending at": see compute_league_rolling_rates'
    docstring for why season S's own games can never legitimately factor
    into season S's own park factor -- a game played in season S needs a
    factor computed from information available before any of season S's
    games happened. A park's very first tracked season -- no prior season
    of its own to roll over at all -- is dropped from the result for the
    same reason (see compute_league_rolling_rates); ParkFactorEmbedding's
    fallback (see index_for) is what a caller gets instead when looking up a
    park's debut season.
    """
    df = season_park_totals.sort_values(["park_id", "season"]).reset_index(drop=True)

    # shift(1) within each park group before rolling: same trick as
    # compute_league_rolling_rates, but per-park so a park never borrows a
    # different park's preceding season, and a park's own first season
    # (nothing to shift in from) comes out NaN rather than 0-or-partial.
    prior = df.groupby("park_id")[_ROLLING_TOTAL_COLS].shift(1)
    rolled = prior.groupby(df["park_id"], group_keys=False).rolling(window=rolling_years, min_periods=1).sum()
    rolled = rolled.reset_index(level=0, drop=True)
    df["rolling_games"] = rolled["n_games"]
    df["hr_rate"] = rolled["total_home_runs"] / rolled["n_games"]
    df["runs_rate"] = rolled["total_runs"] / rolled["n_games"]
    df = df.dropna(subset=["hr_rate", "runs_rate"]).reset_index(drop=True)

    league_rates = compute_league_rolling_rates(season_park_totals, rolling_years)
    df = df.merge(league_rates, on="season", how="inner")
    df["hr_factor"] = df["hr_rate"] / df["league_hr_rate"]
    df["runs_factor"] = df["runs_rate"] / df["league_runs_rate"]

    return df[
        [
            "park_id",
            "season",
            "rolling_games",
            "hr_rate",
            "runs_rate",
            "league_hr_rate",
            "league_runs_rate",
            "hr_factor",
            "runs_factor",
        ]
    ]


def compute_park_factors(pitches: pd.DataFrame, rolling_years: int = DEFAULT_ROLLING_YEARS) -> pd.DataFrame:
    """End-to-end: per-pitch table -> one row per (park_id, season) with
    rolling HR-rate and runs-rate factors relative to league average."""
    game_totals = compute_game_totals(pitches)
    season_totals = compute_season_park_totals(game_totals)
    return compute_rolling_park_factors(season_totals, rolling_years)


def compute_league_rates(pitches: pd.DataFrame, rolling_years: int = DEFAULT_ROLLING_YEARS) -> pd.DataFrame:
    """End-to-end: per-pitch table -> one row per season with the rolling
    league-wide HR rate and runs rate. The continuous counterpart to
    game_dataset.py's post_humidor flag -- meant to be merged onto a game
    table's `season` column (via league_rates_for) as context features for
    Phase 4, not looked up per park."""
    game_totals = compute_game_totals(pitches)
    season_totals = compute_season_park_totals(game_totals)
    return compute_league_rolling_rates(season_totals, rolling_years)


# ---------------------------------------------------------------------------
# Config + learned embedding lookup.
# ---------------------------------------------------------------------------


@dataclass
class ParkFactorConfig:
    rolling_years: int = DEFAULT_ROLLING_YEARS
    embedding_dim: int = DEFAULT_EMBEDDING_DIM

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "ParkFactorConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


class ParkFactorEmbedding(nn.Module):
    """Learned nn.Embedding over (park_id, season) pairs -- a separate,
    independently-trainable row per season a park appears in, rather than
    one static row per park.

    Seeded, not randomly initialized: for embedding_dim >= 2, dimension 0 of
    each real row starts at log(hr_factor) and dimension 1 at
    log(runs_factor) (log so a 2x-inflating park and a 2x-suppressing park
    are symmetric around 0 instead of 1); any remaining dimensions, plus the
    entire reserved unknown-park row, start as small random noise. Every
    dimension -- including the two seeded ones -- is a plain nn.Parameter,
    so a downstream model training against this embedding can move all of
    them; the seeding only sets where each row starts from, not what it's
    limited to.

    Not wired into any model here -- this is Phase 3's deliverable. Phase 4
    (the situational/event model) is expected to call indices_for(...) to
    turn a batch's (park_id, season) pairs into index tensors and then
    ParkFactorEmbedding(indices) to get the corresponding vectors.
    """

    def __init__(self, config: ParkFactorConfig, park_factors: pd.DataFrame) -> None:
        super().__init__()
        self.config = config
        park_factors = park_factors.sort_values(["park_id", "season"]).reset_index(drop=True)

        self.vocab: dict[tuple[str, int], int] = {
            (row.park_id, int(row.season)): i + 1  # +1: row 0 is UNKNOWN_PARK_INDEX
            for i, row in enumerate(park_factors.itertuples(index=False))
        }
        self._seasons_by_park: dict[str, list[int]] = {}
        for park_id, season in self.vocab:
            self._seasons_by_park.setdefault(park_id, []).append(season)
        for seasons in self._seasons_by_park.values():
            seasons.sort()

        num_rows = len(self.vocab) + 1
        self.embedding = nn.Embedding(num_rows, config.embedding_dim)

        init = torch.empty(num_rows, config.embedding_dim)
        nn.init.normal_(init, mean=0.0, std=0.02)
        if config.embedding_dim >= 2:
            # Clipped before logging so a (synthetic-data-only) zero-HR or
            # zero-runs window can't produce -inf; real MLB seasons never
            # hit this floor.
            hr_factor = np.clip(park_factors["hr_factor"].to_numpy(dtype=np.float64), 1e-6, None)
            runs_factor = np.clip(park_factors["runs_factor"].to_numpy(dtype=np.float64), 1e-6, None)
            init[1:, 0] = torch.from_numpy(np.log(hr_factor)).float()
            init[1:, 1] = torch.from_numpy(np.log(runs_factor)).float()
        self.embedding.weight.data.copy_(init)

    def index_for(self, park_id: str, season: int) -> int:
        """(park_id, season) -> row index. An exact match uses that season's
        own row. A known park queried for a season outside its computed
        range falls back to the nearest known season -- the latest one at or
        before it if any exist (e.g. a game in a season not yet fully
        aggregated), otherwise its earliest known season (mirrors
        park_history.py's same both-directions fallback). A park_id with no
        computed rows at all -- never seen in the data this embedding was
        built from -- maps to UNKNOWN_PARK_INDEX."""
        season = int(season)
        key = (park_id, season)
        if key in self.vocab:
            return self.vocab[key]
        seasons = self._seasons_by_park.get(park_id)
        if not seasons:
            return UNKNOWN_PARK_INDEX
        earlier_or_equal = [s for s in seasons if s <= season]
        nearest = max(earlier_or_equal) if earlier_or_equal else min(seasons)
        return self.vocab[(park_id, nearest)]

    def indices_for(self, park_id: pd.Series, season: pd.Series) -> torch.Tensor:
        """Vectorized version of index_for, via a small (park_id, season) ->
        index lookup built from just the unique pairs actually present --
        same approach as park_history.resolve_park_id, for the same reason
        (far fewer unique pairs than rows)."""
        pairs = pd.DataFrame({"park_id": park_id.to_numpy(), "season": season.to_numpy()})
        unique_pairs = pairs.drop_duplicates().reset_index(drop=True)
        unique_pairs["index"] = [
            self.index_for(p, s) for p, s in zip(unique_pairs["park_id"], unique_pairs["season"])
        ]
        merged = pairs.merge(unique_pairs, on=["park_id", "season"], how="left")
        return torch.tensor(merged["index"].to_numpy(), dtype=torch.long)

    def forward(self, index: torch.Tensor) -> torch.Tensor:
        return self.embedding(index)


def build_park_factor_embedding(
    pitches_dir: Path = DEFAULT_PITCHES_DIR, config: ParkFactorConfig | None = None
) -> ParkFactorEmbedding:
    """Main entry point: read the processed per-pitch table, compute park
    factors, and build the embedding lookup from them."""
    config = config or ParkFactorConfig()
    pitches = read_partitioned(pitches_dir)
    park_factors = compute_park_factors(pitches, rolling_years=config.rolling_years)
    return ParkFactorEmbedding(config, park_factors)


if __name__ == "__main__":
    logging_config = ParkFactorConfig.from_yaml()
    factors = compute_park_factors(read_partitioned(DEFAULT_PITCHES_DIR), rolling_years=logging_config.rolling_years)
    latest_season = factors["season"].max()
    latest = factors[factors["season"] == latest_season].sort_values("hr_factor", ascending=False)
    print(f"Park factors as of season {latest_season} (rolling {logging_config.rolling_years}yr window):")
    print(latest.to_string(index=False))
