"""Build a clean, per-plate-appearance feature table from raw Statcast Parquet files.

Same source data and cleaning as build_features.py, but aggregated to one row
per plate appearance (game_pk + at_bat_number) instead of one row per pitch --
the natural unit from the batter's perspective. Flags (without silently
dropping) rows missing any critical field, sorts by batter and then
chronologically, and writes a Parquet dataset partitioned by season to
data/processed/plate_appearances/.

`balls`/`strikes` are the count entering the final (deciding) pitch of the PA,
not the resolved count after it -- e.g. a strikeout looking on a 3-2 pitch is
recorded as strikes=2, matching the raw Statcast convention that the count
reflects the situation the pitch was thrown into, not its result.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data.statcast_common import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    build_pitch_frame,
    discover_raw_seasons,
    flag_missing_critical,
    write_partitioned,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROCESSED_DATA_DIR / "plate_appearances"

CRITICAL_FIELDS = [
    "batter_id",
    "pitcher_id",
    "game_date",
    "game_pk",
    "at_bat_number",
    "inning",
    "outs_when_up",
    "stand",
    "p_throws",
    "home_team",
    "away_team",
    "balls",
    "strikes",
    "outcome",
]


def build_season_plate_appearances_from_frame(pitches: pd.DataFrame) -> pd.DataFrame:
    pitches = pitches.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    grouped = pitches.groupby(["game_pk", "at_bat_number"], sort=False)

    pa = grouped.agg(
        batter_id=("batter_id", "first"),
        pitcher_id=("pitcher_id", "first"),
        game_date=("game_date", "first"),
        season=("season", "first"),
        inning=("inning", "first"),
        outs_when_up=("outs_when_up", "first"),
        stand=("stand", "first"),
        p_throws=("p_throws", "first"),
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        balls=("balls", "last"),
        strikes=("strikes", "last"),
        pitch_count=("pitch_number", "max"),
        # The last pitch of a PA is the one that ends it, so its outcome (an
        # events-derived label -- see compute_outcome) is the PA's outcome.
        outcome=("outcome", "last"),
    ).reset_index()

    pa["is_valid"], pa["missing_fields"] = flag_missing_critical(pa, CRITICAL_FIELDS)
    return pa.sort_values(["batter_id", "game_date", "at_bat_number"]).reset_index(drop=True)


def build_season_plate_appearances(raw_path: Path) -> pd.DataFrame:
    return build_season_plate_appearances_from_frame(build_pitch_frame(raw_path))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the per-plate-appearance feature table from raw Statcast data."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    raw_files = discover_raw_seasons(args.raw_dir)
    if not raw_files:
        logger.warning("No raw Statcast files found in %s", args.raw_dir)
        return

    for raw_path in raw_files:
        logger.info("Building plate-appearance table for %s", raw_path.name)
        season_df = build_season_plate_appearances(raw_path)
        write_partitioned(season_df, args.output_dir)

    logger.info("Done. Wrote %d season(s) to %s", len(raw_files), args.output_dir)


if __name__ == "__main__":
    main()
