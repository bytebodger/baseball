"""Build the betting_lines table by joining a third-party game-level odds
export (data/raw/MLB_Basic.csv) onto the existing games table
(data/processed/games) so each betting line is keyed by game_pk.

The source file is a flat CSV with one row per game: game id, date
(YYYYMMDD), away/home team codes, away/home final score, and open/close
moneyline + total(over/under) odds. It uses several team-abbreviation
variants the Statcast pipeline doesn't (see configs/betting_lines.yaml for
the full crosswalk and why each variant exists), and it has no game_pk of
its own -- the only way to identify which Statcast game a row refers to is
to match on (date, home_team, away_team) after normalizing team codes.

That match is only trusted when it's unambiguous *and* verified:
  - Team codes not in the crosswalk (e.g. "AL"/"NL" All-Star Game rows)
    can't be normalized at all -- sent to review.
  - If either side has more than one row for the same (date, home, away) key
    (a doubleheader), the match is ambiguous -- sent to review rather than
    guessed at via score-matching.
  - Otherwise the single candidate match is accepted only if away_score and
    home_score match exactly between the two sources; a mismatch means the
    date/team key lined up by coincidence (or one source has a data error),
    so it's sent to review instead of accepted silently.

Only seasons already present in data/processed/games are considered --
this script does not build new game-table seasons as a side effect. CSV
rows from seasons not yet built (2009-2014) are sent to review with a
reason that says so, rather than being silently dropped.
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.statcast_common import PROCESSED_DATA_DIR, RAW_DATA_DIR, read_partitioned

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_CSV_PATH = RAW_DATA_DIR / "MLB_Basic.csv"
GAMES_DIR = PROCESSED_DATA_DIR / "games"
OUTPUT_DIR = PROCESSED_DATA_DIR / "betting_lines"
REVIEW_PATH = PROCESSED_DATA_DIR / "betting_lines_review.csv"
CROSSWALK_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "betting_lines.yaml"

JOIN_KEY = ["game_date", "home_team", "away_team"]

# CSV column -> working column name. "over open"/"under open" (and the close
# equivalents) are always identical in this file (the total line, quoted
# once for each side of the bet) -- verified in _read_betting_csv rather than
# assumed, since a future refresh of the file isn't guaranteed to hold.
_CSV_COLUMN_RENAME = {
    "game id": "source_game_id",
    "date": "raw_date",
    "away team": "raw_away_team",
    "away score": "away_score",
    "away ml open": "away_ml_open",
    "away ml close": "away_ml_close",
    "over open": "total_open",
    "over open odds": "over_open_odds",
    "over close": "total_close",
    "over close odds": "over_close_odds",
    "home team": "raw_home_team",
    "home score": "home_score",
    "home ml open": "home_ml_open",
    "home ml close": "home_ml_close",
    "under open": "_under_open",
    "under open odds": "under_open_odds",
    "under close": "_under_close",
    "under close odds": "under_close_odds",
}

BETTING_LINE_COLUMNS = [
    "game_pk", "game_date", "season", "home_team", "away_team",
    "away_ml_open", "away_ml_close", "home_ml_open", "home_ml_close",
    "total_open", "over_open_odds", "under_open_odds",
    "total_close", "over_close_odds", "under_close_odds",
]

REVIEW_COLUMNS = [
    "reason", "source_game_id", "raw_date", "raw_away_team", "raw_home_team",
    "away_score", "home_score",
]


def load_team_crosswalk(config_path: Path = CROSSWALK_CONFIG_PATH) -> dict[str, str]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["team_abbreviation_crosswalk"]


def american_odds_to_implied_prob(odds) -> np.ndarray:
    """Converts American moneyline odds to each side's raw implied probability
    (favorite/underdog, no adjustment for the vig). Positive odds (underdog):
    100 / (odds + 100). Negative odds (favorite): -odds / (-odds + 100)."""
    odds = np.asarray(odds, dtype=float)
    # np.where evaluates both branches for every element, so e.g. odds == -100
    # triggers a spurious divide-by-zero in the unselected positive-odds branch
    # (and vice versa for odds == 100) even though the actual result is fine --
    # suppress just that warning rather than the result being wrong.
    with np.errstate(divide="ignore"):
        return np.where(odds > 0, 100.0 / (odds + 100.0), -odds / (-odds + 100.0))


def no_vig_home_win_prob(home_ml, away_ml) -> np.ndarray:
    """Converts both sides' moneyline odds into a de-vigged home-team win
    probability: each side's raw implied probability sums to over 1.0 (the
    sportsbook's overround/vig), so normalizing both by that sum removes it
    and gives a pair of probabilities that sum to exactly 1.0."""
    home_implied = american_odds_to_implied_prob(home_ml)
    away_implied = american_odds_to_implied_prob(away_ml)
    return home_implied / (home_implied + away_implied)


def _normalize_team_token(raw_token: str) -> str | None:
    """Strip whitespace and any stray trailing punctuation (e.g. the "CIN -"
    typo present in a couple of rows) down to the leading run of letters."""
    match = re.match(r"[A-Za-z]+", raw_token.strip())
    return match.group(0).upper() if match else None


def _read_betting_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns=_CSV_COLUMN_RENAME)

    mismatched_total = (df["total_open"] != df["_under_open"]) | (df["total_close"] != df["_under_close"])
    if mismatched_total.any():
        logger.warning(
            "%d row(s) have a different over/under total than under/over -- keeping the 'over' total.",
            int(mismatched_total.sum()),
        )
    df = df.drop(columns=["_under_open", "_under_close"])

    df["game_date"] = pd.to_datetime(df["raw_date"], format="%Y%m%d")
    df["away_team"] = df["raw_away_team"].map(_normalize_team_token)
    df["home_team"] = df["raw_home_team"].map(_normalize_team_token)
    return df


def _apply_crosswalk(df: pd.DataFrame, crosswalk: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["away_team"] = df["away_team"].map(crosswalk)
    df["home_team"] = df["home_team"].map(crosswalk)
    return df


def _to_review_rows(df: pd.DataFrame, reason: str) -> pd.DataFrame:
    review = df.reindex(columns=REVIEW_COLUMNS[1:]).copy()
    review.insert(0, "reason", reason)
    return review


def build_betting_lines(betting_csv: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `betting_csv` rows into (accepted betting_lines rows, review rows).

    `betting_csv` is expected to already have normalized `away_team`/`home_team`
    columns (NaN where the raw code had no crosswalk entry). `games` is the
    existing games table (game_pk, game_date, season, home_team, away_team,
    home_score, away_score, ...) restricted to whatever seasons the caller
    wants matched against.
    """
    review_frames = []

    unmapped = betting_csv["away_team"].isna() | betting_csv["home_team"].isna()
    if unmapped.any():
        review_frames.append(_to_review_rows(betting_csv[unmapped], "unmapped team code"))
    mapped = betting_csv[~unmapped].copy()

    built_seasons = set(games["season"].unique())
    season_of_row = mapped["game_date"].dt.year
    not_built = ~season_of_row.isin(built_seasons)
    if not_built.any():
        for season, rows in mapped[not_built].groupby(season_of_row[not_built]):
            review_frames.append(_to_review_rows(rows, f"season {season} not present in games table"))
    mapped = mapped[~not_built]

    csv_key_counts = mapped.groupby(JOIN_KEY)[JOIN_KEY[0]].transform("size")
    csv_ambiguous = csv_key_counts > 1
    if csv_ambiguous.any():
        review_frames.append(_to_review_rows(mapped[csv_ambiguous], "doubleheader (multiple betting-line rows for this date/matchup)"))
    csv_unique = mapped[~csv_ambiguous]

    games_key_counts = games.groupby(JOIN_KEY)[JOIN_KEY[0]].transform("size")
    games_unique = games[games_key_counts == 1]
    games_ambiguous_keys = set(map(tuple, games.loc[games_key_counts > 1, JOIN_KEY].itertuples(index=False, name=None)))

    is_ambiguous_on_games_side = csv_unique[JOIN_KEY].apply(tuple, axis=1).isin(games_ambiguous_keys)
    if is_ambiguous_on_games_side.any():
        review_frames.append(
            _to_review_rows(csv_unique[is_ambiguous_on_games_side], "doubleheader (multiple games in games table for this date/matchup)")
        )
    csv_candidates = csv_unique[~is_ambiguous_on_games_side]

    merged = csv_candidates.merge(
        games_unique[["game_pk", "season", "home_score", "away_score"] + JOIN_KEY],
        on=JOIN_KEY,
        how="left",
        suffixes=("", "_games"),
    )

    no_match = merged["game_pk"].isna()
    if no_match.any():
        review_frames.append(_to_review_rows(merged[no_match], "no matching game found"))
    matched = merged[~no_match].copy()

    score_mismatch = (matched["away_score"] != matched["away_score_games"]) | (matched["home_score"] != matched["home_score_games"])
    if score_mismatch.any():
        review_frames.append(_to_review_rows(matched[score_mismatch], "score mismatch vs games table"))
    accepted = matched[~score_mismatch].copy()
    accepted["game_pk"] = accepted["game_pk"].astype("int64")
    accepted["season"] = accepted["season"].astype("int64")

    betting_lines = accepted[BETTING_LINE_COLUMNS].reset_index(drop=True)
    review = (
        pd.concat(review_frames, ignore_index=True)[REVIEW_COLUMNS]
        if review_frames
        else pd.DataFrame(columns=REVIEW_COLUMNS)
    )
    return betting_lines, review


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the betting_lines table from MLB_Basic.csv.")
    parser.add_argument("--csv-path", type=Path, default=RAW_CSV_PATH)
    parser.add_argument("--games-dir", type=Path, default=GAMES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--review-path", type=Path, default=REVIEW_PATH)
    parser.add_argument("--crosswalk-config", type=Path, default=CROSSWALK_CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    crosswalk = load_team_crosswalk(args.crosswalk_config)
    betting_csv = _read_betting_csv(args.csv_path)
    betting_csv = _apply_crosswalk(betting_csv, crosswalk)

    games = read_partitioned(args.games_dir)[["game_pk", "game_date", "season", "home_team", "away_team", "home_score", "away_score"]]

    betting_lines, review = build_betting_lines(betting_csv, games)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    betting_lines.to_parquet(args.output_dir, partition_cols=["season"], index=False)
    review.to_csv(args.review_path, index=False)

    logger.info(
        "Wrote %d betting_lines row(s) to %s; %d row(s) sent to review at %s",
        len(betting_lines), args.output_dir, len(review), args.review_path,
    )


if __name__ == "__main__":
    main()
