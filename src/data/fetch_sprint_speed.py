"""Pulls each player's seasonal sprint speed from Baseball Savant's
sprint-speed leaderboard (via pybaseball) and writes it as its own small
processed table, keyed by (batter_id, season) -- data/processed/sprint_speed.

This is the same "separate table, joined at read time" pattern
build_betting_lines.py uses for Vegas odds, rather than merging sprint speed
destructively into data/processed/batter_appearances. batter_appearances is
a build artifact of game_dataset.py's ensure_game_tables_built, which decides
whether to (re)build a season's games/pitcher_appearances/batter_appearances
together purely by checking whether that season's games/ partition already
exists -- writing sprint speed directly onto batter_appearances would get
silently blown away the next time that season is rebuilt with force=True (or
just never built yet). A standalone table has no such staleness trap:
whoever needs sprint speed on a batter-keyed table joins this one on by
(batter_id, season) whenever they read it.
"""

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd
from pybaseball import statcast_sprint_speed

from src.data.statcast_common import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROCESSED_DATA_DIR / "sprint_speed"
DEFAULT_START_YEAR = 2015
DEFAULT_MIN_OPPORTUNITIES = 10


def fetch_season_sprint_speed(year: int, min_opportunities: int = DEFAULT_MIN_OPPORTUNITIES) -> pd.DataFrame:
    """One row per player who qualified for `year`'s sprint-speed leaderboard.
    `player_id` is renamed to `batter_id` to match this project's naming
    convention (see sequence_dataset.py, statcast_common.py)."""
    raw = statcast_sprint_speed(year, min_opp=min_opportunities)
    return pd.DataFrame(
        {
            "batter_id": raw["player_id"],
            "season": year,
            "sprint_speed": raw["sprint_speed"],
        }
    )


def fetch_and_save_season(year: int, output_dir: Path = OUTPUT_DIR, min_opportunities: int = DEFAULT_MIN_OPPORTUNITIES, force: bool = False) -> None:
    """Fetches one season's sprint-speed leaderboard and appends it to the
    (batter_id, season) -> sprint_speed table at `output_dir`, unless that
    season's partition is already there."""
    season_dir = output_dir / f"season={year}"
    if season_dir.exists() and not force:
        logger.info("Season %d already pulled, skipping (%s)", year, season_dir)
        return

    logger.info("Fetching sprint-speed leaderboard for %d", year)
    season_df = fetch_season_sprint_speed(year, min_opportunities)
    if season_df.empty:
        logger.warning("No sprint-speed rows returned for %d, nothing written", year)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    season_df.to_parquet(output_dir, partition_cols=["season"], index=False)
    logger.info("Saved %d players' sprint speed for %d to %s", len(season_df), year, output_dir)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull seasonal sprint-speed leaderboards from Baseball Savant and write them as a "
        "standalone (batter_id, season) -> sprint_speed table."
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--min-opportunities", type=int, default=DEFAULT_MIN_OPPORTUNITIES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Re-pull a season even if it's already in --output-dir.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    for year in range(args.start_year, args.end_year + 1):
        fetch_and_save_season(year, args.output_dir, args.min_opportunities, args.force)


if __name__ == "__main__":
    main()
