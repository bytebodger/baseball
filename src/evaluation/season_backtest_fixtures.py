"""Shared game-selection and per-game fixture-building for the boundary-2
full-season simulation-based backtest (Phase 11 scoping): every real
TEST_SEASON_RANGE game that also has real Vegas betting lines (required for
the CLV/ROI legs of the backtest) becomes one row of the sample. Used by
both the embedding-cache-extension step and the simulation step so the two
can never quietly disagree about which games/players/dates are in scope.

Generalizes the fixture-building approach already validated by the 35-game
low-scoring-game calibration probe (this session's scratchpad,
low_scoring_calibration.py) to every eligible game in a season, not a
filtered sample of it.
"""

from __future__ import annotations

import pandas as pd

from src.data.game_dataset import BATTER_APPEARANCES_DIR, DEFAULT_BULLPEN_WINDOW_DAYS, GAMES_DIR, PITCHER_APPEARANCES_DIR
from src.data.statcast_common import PROCESSED_DATA_DIR, read_partitioned


def select_season_games_with_betting_lines(season: int, games_dir=GAMES_DIR, betting_lines_dir=None) -> pd.DataFrame:
    """One row per real `season` game that also has real betting-line
    coverage (inner join on game_pk) -- betting lines aren't available for
    every game (2,414/2,477 for the 2025 season, as of 2026-07), and a game
    without lines can't contribute to the CLV/ROI legs of this backtest, so
    it's excluded from the whole sample rather than included with NaN
    CLV/ROI (which would silently skew Brier/log-loss/calibration toward a
    different, larger population than the betting-strategy metrics)."""
    betting_lines_dir = betting_lines_dir or (PROCESSED_DATA_DIR / "betting_lines")
    games = read_partitioned(games_dir)
    season_games = games[games["season"] == season].reset_index(drop=True)
    lines = read_partitioned(betting_lines_dir)
    joined = season_games.merge(lines, on="game_pk", how="inner", suffixes=("", "_lines"))
    return joined.sort_values("game_pk").reset_index(drop=True)


def build_fixture(
    game_row: pd.Series,
    batter_appearances: pd.DataFrame,
    pitcher_appearances: pd.DataFrame,
    bullpen_window_days: int = DEFAULT_BULLPEN_WINDOW_DAYS,
) -> dict:
    """Same construction as the low-scoring-game probe's build_fixture: real
    starters, real lineup (batting order via first_at_bat_number), real
    bullpen (any pitcher who appeared for that team in the trailing
    `bullpen_window_days` before this game, excluding the starter)."""
    game_pk = int(game_row["game_pk"])
    game_date = pd.Timestamp(game_row["game_date"])
    home_team, away_team = game_row["home_team"], game_row["away_team"]
    home_starter, away_starter = int(game_row["home_starter_id"]), int(game_row["away_starter_id"])

    game_batters = batter_appearances[batter_appearances["game_pk"] == game_pk]

    def lineup_for(team):
        team_batters = game_batters[game_batters["team"] == team].sort_values("first_at_bat_number")
        return team_batters["batter_id"].astype(int).tolist()[:9]

    window_start = game_date - pd.Timedelta(days=bullpen_window_days)

    def bullpen_for(team, exclude_starter):
        window = pitcher_appearances[
            (pitcher_appearances["team"] == team)
            & (pitcher_appearances["game_date"] >= window_start)
            & (pitcher_appearances["game_date"] < game_date)
        ]
        return sorted(set(window["pitcher_id"].astype(int).tolist()) - {exclude_starter})

    return {
        "game_pk": game_pk,
        "game_date": str(game_date.date()),
        "home_team": home_team,
        "away_team": away_team,
        "home_starter": home_starter,
        "away_starter": away_starter,
        "home_lineup": lineup_for(home_team),
        "away_lineup": lineup_for(away_team),
        "home_bullpen": bullpen_for(home_team, home_starter),
        "away_bullpen": bullpen_for(away_team, away_starter),
        "actual_home_score": int(game_row["home_score"]),
        "actual_away_score": int(game_row["away_score"]),
        "actual_home_win": bool(game_row["home_win"]),
    }


def load_appearance_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    return read_partitioned(BATTER_APPEARANCES_DIR), read_partitioned(PITCHER_APPEARANCES_DIR)


def pitcher_and_batter_query_pairs(fixtures: list[dict]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Every distinct (pitcher_id, game_date_ns) / (batter_id, game_date_ns)
    pair simulate_games_batch will need an embedding for, across every
    fixture -- deduplicated, for extending the embedding cache exactly far
    enough to cover this specific game sample (not a broader/looser
    'every roster-active date' sweep, which this sample doesn't need)."""
    pitcher_pairs: set[tuple[int, int]] = set()
    batter_pairs: set[tuple[int, int]] = set()
    for fx in fixtures:
        date_ns = pd.Timestamp(fx["game_date"]).value
        for pid in [fx["home_starter"], fx["away_starter"], *fx["home_bullpen"], *fx["away_bullpen"]]:
            pitcher_pairs.add((pid, date_ns))
        for bid in [*fx["home_lineup"], *fx["away_lineup"]]:
            batter_pairs.add((bid, date_ns))
    return sorted(pitcher_pairs), sorted(batter_pairs)
