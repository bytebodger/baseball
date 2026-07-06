"""GameOutcomeDataset: gathers everything needed to predict a game's outcome
from pre-game pitch/plate-appearance history.

For each historical game this returns the home and away starting pitchers'
pitch histories, both bullpens' available relievers, both lineups' batters,
context features, and the actual final score / win-loss label. All player
histories use PlayerPitchSequenceDataset with cutoff=game_date, so a player's
sequence for a given game never includes a pitch from that game or any later
game -- the "strictly before" filter in PlayerPitchSequenceDataset is what
guarantees no game-level leakage.

A few things worth knowing about how this is built, since they're genuine
design decisions rather than obvious defaults:

- Starting pitcher and starting lineup are derived from Statcast itself (who
  actually threw the game's first pitch for each side; the first 9 distinct
  batters to reach the plate for each side, in order), not from a separate
  probable-starters feed. Real starting lineups/pitchers are announced well
  before first pitch, so this is safe, pre-game-knowable information -- it's
  just sourced from the box score after the fact rather than a live schedule
  API. Restricting the lineup to the first 9 distinct batters (rather than
  every batter who ever appeared) also excludes pinch hitters/subs, who would
  otherwise leak in-game situational information (why was a pinch hitter
  used?) into a pre-game feature.
- "Available bullpen" is a trailing-window proxy: pitchers who appeared for
  that team in any game in the `bullpen_window_days` days strictly before the
  game (default 14), excluding that game's own starter. This is deliberately
  NOT "who actually relieved in this game" -- which reliever a team actually
  used reveals in-game information (mop-up relievers imply a blowout) that
  wouldn't be known before the game. There's no clean historical "26-man
  active roster as of date X" data source behind pybaseball for arbitrary
  past seasons, so this proxy is used instead of a live roster/schedule
  lookup; it approximates roster availability from a team's own recent
  Statcast usage patterns.
- `park_id` is just `home_team`, since we don't have a separate venue table
  and teams play the vast majority of their "home" games at one park --
  same spirit as using month-of-season as a weather proxy.

Season split matches pretrain_encoder.py exactly (same shared constants from
statcast_common): train 2015-2022, validate 2023, 2024-2025 held out.

Only regular-season and postseason games (`game_type` in R/F/D/L/W) count as
"historical games" here. fetch_statcast.py pulls full March-November date
ranges, which also sweeps in spring training ('S') games -- those are
filtered out when building the games table, since spring training rosters
and performance aren't representative of the real season.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.sequence_dataset import PlayerPitchSequenceDataset
from src.data.statcast_common import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TRAIN_SEASON_RANGE,
    VAL_SEASONS,
    read_partitioned,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GAMES_DIR = PROCESSED_DATA_DIR / "games"
PITCHER_APPEARANCES_DIR = PROCESSED_DATA_DIR / "pitcher_appearances"
BATTER_APPEARANCES_DIR = PROCESSED_DATA_DIR / "batter_appearances"

DEFAULT_BULLPEN_WINDOW_DAYS = 14
DEFAULT_MAX_LINEUP_SIZE = 9

_RAW_COLUMNS = [
    "game_pk", "game_date", "game_year", "game_type", "home_team", "away_team",
    "inning_topbot", "inning", "at_bat_number", "pitch_number",
    "pitcher", "batter", "post_home_score", "post_away_score",
]

# fetch_statcast.py pulls full March-November date ranges, which sweeps in
# spring training ('S') games alongside regular season ('R') and postseason
# ('F' wild card, 'D' division series, 'L' championship series, 'W' World
# Series). Spring training rosters/performance aren't representative of the
# real season, so they're excluded here rather than in the raw pull itself.
_REAL_GAME_TYPES = {"R", "F", "D", "L", "W"}


def _build_season_game_tables(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """From one season's raw Statcast file, derive:
    - games: one row per game_pk (score, home_win, month, starters, rest days)
    - pitcher_appearances: one row per (game_pk, team, pitcher) with is_starter
    - batter_appearances: one row per (game_pk, team, batter) with their first
      at_bat_number in the game, used to reconstruct the starting lineup order
    """
    raw = pd.read_parquet(raw_path, columns=_RAW_COLUMNS)
    raw = raw[raw["game_type"].isin(_REAL_GAME_TYPES)].drop(columns="game_type")
    raw["game_date"] = pd.to_datetime(raw["game_date"])

    # The pitcher fields the defensive team; the batter fields the offensive
    # team. inning_topbot tells us which side is which.
    is_top = raw["inning_topbot"] == "Top"  # away team batting, home team pitching
    raw["pitcher_team"] = np.where(is_top, raw["home_team"], raw["away_team"])
    raw["batter_team"] = np.where(is_top, raw["away_team"], raw["home_team"])

    order_cols = ["game_pk", "pitcher_team", "inning", "at_bat_number", "pitch_number"]
    sorted_raw = raw.sort_values(order_cols)
    starters = sorted_raw.drop_duplicates(subset=["game_pk", "pitcher_team"], keep="first")[
        ["game_pk", "pitcher_team", "pitcher", "home_team", "away_team"]
    ].rename(columns={"pitcher": "starter_id"})

    pitcher_appearances = raw[["game_pk", "pitcher_team", "pitcher", "game_date", "game_year"]].drop_duplicates(
        subset=["game_pk", "pitcher_team", "pitcher"]
    ).rename(columns={"pitcher_team": "team", "pitcher": "pitcher_id", "game_year": "season"})
    pitcher_appearances = pitcher_appearances.merge(
        starters[["game_pk", "pitcher_team", "starter_id"]].rename(columns={"pitcher_team": "team"}),
        on=["game_pk", "team"],
        how="left",
    )
    pitcher_appearances["is_starter"] = pitcher_appearances["pitcher_id"] == pitcher_appearances["starter_id"]
    pitcher_appearances = pitcher_appearances.drop(columns="starter_id").reset_index(drop=True)

    batter_appearances = raw.groupby(["game_pk", "batter_team", "batter"], as_index=False).agg(
        game_date=("game_date", "first"),
        season=("game_year", "first"),
        first_at_bat_number=("at_bat_number", "min"),
    ).rename(columns={"batter_team": "team", "batter": "batter_id"})

    games = raw.groupby("game_pk", as_index=False).agg(
        game_date=("game_date", "first"),
        season=("game_year", "first"),
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        home_score=("post_home_score", "max"),
        away_score=("post_away_score", "max"),
    )
    games["home_win"] = games["home_score"] > games["away_score"]
    games["month"] = games["game_date"].dt.month

    side = np.where(starters["pitcher_team"] == starters["home_team"], "home", "away")
    home_starters = starters[side == "home"][["game_pk", "starter_id"]].rename(columns={"starter_id": "home_starter_id"})
    away_starters = starters[side == "away"][["game_pk", "starter_id"]].rename(columns={"starter_id": "away_starter_id"})
    games = games.merge(home_starters, on="game_pk", how="left").merge(away_starters, on="game_pk", how="left")

    starts_only = pitcher_appearances[pitcher_appearances["is_starter"]].sort_values(["pitcher_id", "game_date"])
    rest_days = starts_only.groupby("pitcher_id")["game_date"].diff().dt.days
    rest_for_join = starts_only[["game_pk", "pitcher_id"]].assign(rest_days=rest_days)

    games = games.merge(
        rest_for_join.rename(columns={"pitcher_id": "home_starter_id", "rest_days": "home_starter_rest_days"}),
        on=["game_pk", "home_starter_id"],
        how="left",
    ).merge(
        rest_for_join.rename(columns={"pitcher_id": "away_starter_id", "rest_days": "away_starter_rest_days"}),
        on=["game_pk", "away_starter_id"],
        how="left",
    )

    before = len(games)
    games = games.dropna(
        subset=["home_starter_id", "away_starter_id", "home_score", "away_score"]
    ).reset_index(drop=True)
    dropped = before - len(games)
    if dropped:
        logger.info(
            "Dropped %d/%d incomplete games (missing starter or score) in %s", dropped, before, raw_path.name
        )

    return games, pitcher_appearances, batter_appearances


def ensure_game_tables_built(
    seasons: list[int],
    raw_dir: Path = RAW_DATA_DIR,
    games_dir: Path = GAMES_DIR,
    pitcher_appearances_dir: Path = PITCHER_APPEARANCES_DIR,
    batter_appearances_dir: Path = BATTER_APPEARANCES_DIR,
    force: bool = False,
) -> None:
    """Build and cache games/pitcher_appearances/batter_appearances for any of
    `seasons` not already on disk (or all of them, if force=True)."""
    for season in seasons:
        if not force and (games_dir / f"season={season}").exists():
            continue

        raw_path = raw_dir / f"statcast_{season}.parquet"
        logger.info("Building game tables for season %d from %s", season, raw_path.name)
        games, pitcher_appearances, batter_appearances = _build_season_game_tables(raw_path)

        games.to_parquet(games_dir, partition_cols=["season"], index=False)
        pitcher_appearances.to_parquet(pitcher_appearances_dir, partition_cols=["season"], index=False)
        batter_appearances.to_parquet(batter_appearances_dir, partition_cols=["season"], index=False)


def load_game_split(
    raw_dir: Path = RAW_DATA_DIR,
    games_dir: Path = GAMES_DIR,
    pitcher_appearances_dir: Path = PITCHER_APPEARANCES_DIR,
    batter_appearances_dir: Path = BATTER_APPEARANCES_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ensure games/pitcher_appearances/batter_appearances are built for every
    season in TRAIN_SEASON_RANGE + VAL_SEASONS (2024-2025 is never touched here),
    then return (train_games, val_games, pitcher_appearances, batter_appearances).

    pitcher_appearances/batter_appearances are returned covering the full
    train+val season range regardless of split -- that's safe (not leakage)
    because PlayerPitchSequenceDataset's own "strictly before cutoff" filter
    is what actually enforces no-leakage per game, not how much data these
    lookup tables happen to span.
    """
    seasons = list(range(TRAIN_SEASON_RANGE[0], TRAIN_SEASON_RANGE[1] + 1)) + list(VAL_SEASONS)
    ensure_game_tables_built(seasons, raw_dir, games_dir, pitcher_appearances_dir, batter_appearances_dir)

    games = read_partitioned(games_dir)
    pitcher_appearances = read_partitioned(pitcher_appearances_dir)
    batter_appearances = read_partitioned(batter_appearances_dir)

    train_games = games[games["season"].between(*TRAIN_SEASON_RANGE)].reset_index(drop=True)
    val_games = games[games["season"].isin(VAL_SEASONS)].reset_index(drop=True)
    return train_games, val_games, pitcher_appearances, batter_appearances


class GameOutcomeDataset(Dataset):
    """One sample per historical game. `pitches` must already be filtered to
    `is_valid` rows (see build_features.py) -- same expectation as
    NextPitchDataset in pretrain_encoder.py.
    """

    def __init__(
        self,
        pitches: pd.DataFrame,
        games: pd.DataFrame,
        pitcher_appearances: pd.DataFrame,
        batter_appearances: pd.DataFrame,
        max_seq_len: int,
        bullpen_window_days: int = DEFAULT_BULLPEN_WINDOW_DAYS,
        max_lineup_size: int = DEFAULT_MAX_LINEUP_SIZE,
        continuous_stats: dict[str, tuple[float, float]] | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.games = games.reset_index(drop=True)
        self.bullpen_window = pd.Timedelta(days=bullpen_window_days)
        self.max_lineup_size = max_lineup_size

        continuous_stats = continuous_stats or PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
        self.pitcher_sequences = PlayerPitchSequenceDataset(
            pitches, samples=[], max_seq_len=max_seq_len, perspective="pitcher",
            continuous_stats=continuous_stats, cache_dir=cache_dir,
        )
        self.batter_sequences = PlayerPitchSequenceDataset(
            pitches, samples=[], max_seq_len=max_seq_len, perspective="batter",
            continuous_stats=continuous_stats, cache_dir=cache_dir,
        )

        self._appearances_by_team = {
            team: group.sort_values("game_date").reset_index(drop=True)
            for team, group in pitcher_appearances.groupby("team")
        }
        self._lineup_by_game_team = {
            key: group.sort_values("first_at_bat_number")["batter_id"].tolist()
            for key, group in batter_appearances.groupby(["game_pk", "team"])
        }

    def __len__(self) -> int:
        return len(self.games)

    def _bullpen_ids(self, team: str, cutoff: pd.Timestamp, exclude_id) -> list[int]:
        team_appearances = self._appearances_by_team.get(team)
        if team_appearances is None:
            return []
        window_start = cutoff - self.bullpen_window
        mask = (team_appearances["game_date"] >= window_start) & (team_appearances["game_date"] < cutoff)
        ids = team_appearances.loc[mask, "pitcher_id"].unique().tolist()
        return [pid for pid in ids if pid != exclude_id]

    def _lineup_ids(self, game_pk: int, team: str) -> list[int]:
        return self._lineup_by_game_team.get((game_pk, team), [])[: self.max_lineup_size]

    def warm_cache(self) -> tuple[int, int]:
        """Precomputes and disk-caches every player sequence this dataset's
        games will ever ask for -- both starters, every trailing-window
        bullpen arm, every lineup batter, for every game in self.games --
        so __getitem__ never has to build one from scratch during training.
        Requires cache_dir to have been set on this dataset.

        Call this once, single-process, before handing the dataset to a
        DataLoader (especially one using multiple worker processes -- see
        PlayerPitchSequenceDataset.precompute_and_cache for why the cache is
        write-once-up-front rather than written lazily during __getitem__).

        Returns (pitcher_sequences_computed, batter_sequences_computed) --
        i.e. how many were actually new work, not already cached from a
        previous run against the same cache_dir.
        """
        pitcher_queries = []
        batter_queries = []

        for game in self.games.itertuples():
            cutoff = pd.Timestamp(game.game_date)

            pitcher_queries.append((game.home_starter_id, cutoff))
            pitcher_queries.append((game.away_starter_id, cutoff))
            for pid in self._bullpen_ids(game.home_team, cutoff, game.home_starter_id):
                pitcher_queries.append((pid, cutoff))
            for pid in self._bullpen_ids(game.away_team, cutoff, game.away_starter_id):
                pitcher_queries.append((pid, cutoff))

            for bid in self._lineup_ids(game.game_pk, game.home_team):
                batter_queries.append((bid, cutoff))
            for bid in self._lineup_ids(game.game_pk, game.away_team):
                batter_queries.append((bid, cutoff))

        pitcher_computed = self.pitcher_sequences.precompute_and_cache(pitcher_queries)
        batter_computed = self.batter_sequences.precompute_and_cache(batter_queries)
        return pitcher_computed, batter_computed

    def __getitem__(self, idx: int) -> dict:
        game = self.games.iloc[idx]
        cutoff = pd.Timestamp(game["game_date"])
        home_starter_id = game["home_starter_id"]
        away_starter_id = game["away_starter_id"]

        home_bullpen_ids = self._bullpen_ids(game["home_team"], cutoff, home_starter_id)
        away_bullpen_ids = self._bullpen_ids(game["away_team"], cutoff, away_starter_id)
        home_lineup_ids = self._lineup_ids(game["game_pk"], game["home_team"])
        away_lineup_ids = self._lineup_ids(game["game_pk"], game["away_team"])

        return {
            "game_pk": int(game["game_pk"]),
            "game_date": cutoff,
            "season": int(game["season"]),
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "park_id": game["home_team"],
            "month": cutoff.month,
            "home_starter_rest_days": game["home_starter_rest_days"],
            "away_starter_rest_days": game["away_starter_rest_days"],
            "home_starter": self.pitcher_sequences.build_sequence(home_starter_id, cutoff),
            "away_starter": self.pitcher_sequences.build_sequence(away_starter_id, cutoff),
            "home_bullpen": [self.pitcher_sequences.build_sequence(pid, cutoff) for pid in home_bullpen_ids],
            "away_bullpen": [self.pitcher_sequences.build_sequence(pid, cutoff) for pid in away_bullpen_ids],
            "home_lineup": [self.batter_sequences.build_sequence(bid, cutoff) for bid in home_lineup_ids],
            "away_lineup": [self.batter_sequences.build_sequence(bid, cutoff) for bid in away_lineup_ids],
            "home_score": int(game["home_score"]),
            "away_score": int(game["away_score"]),
            "home_win": bool(game["home_win"]),
        }


def load_train_val_game_datasets(
    pitches_dir: Path = PROCESSED_DATA_DIR / "pitches",
    max_seq_len: int = 200,
    bullpen_window_days: int = DEFAULT_BULLPEN_WINDOW_DAYS,
    max_lineup_size: int = DEFAULT_MAX_LINEUP_SIZE,
) -> tuple["GameOutcomeDataset", "GameOutcomeDataset"]:
    """Convenience constructor: loads the season-split pitches + game tables
    and returns (train_dataset, val_dataset), sharing one set of continuous
    feature normalization stats computed from the training pitches only."""
    full_pitches = read_partitioned(pitches_dir)
    pitches = full_pitches[
        full_pitches["season"].between(TRAIN_SEASON_RANGE[0], VAL_SEASONS[-1]) & full_pitches["is_valid"]
    ].reset_index(drop=True)
    train_pitches = pitches[pitches["season"].between(*TRAIN_SEASON_RANGE)]
    continuous_stats = PlayerPitchSequenceDataset._compute_continuous_stats(train_pitches)

    train_games, val_games, pitcher_appearances, batter_appearances = load_game_split()

    train_dataset = GameOutcomeDataset(
        pitches, train_games, pitcher_appearances, batter_appearances,
        max_seq_len, bullpen_window_days, max_lineup_size, continuous_stats,
    )
    val_dataset = GameOutcomeDataset(
        pitches, val_games, pitcher_appearances, batter_appearances,
        max_seq_len, bullpen_window_days, max_lineup_size, continuous_stats,
    )
    return train_dataset, val_dataset
