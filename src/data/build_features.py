"""Build a clean, per-pitch feature table from raw Statcast Parquet files.

Reads data/raw/statcast_{year}.parquet (one file per season), selects and
renames the modeling columns, encodes each pitch's result into a single
`outcome` label, flags (without silently dropping) rows missing any critical
field, sorts by pitcher and then chronologically, and writes the result as a
Parquet dataset partitioned by season to data/processed/pitches/.
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

OUTPUT_DIR = PROCESSED_DATA_DIR / "pitches"

CRITICAL_FIELDS = [
    "pitcher_id",
    "batter_id",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitch_type",
    "release_speed",
    "spin_rate",
    "plate_x",
    "plate_z",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "stand",
    "p_throws",
    "home_team",
    "away_team",
    "outcome",
]


def build_season_pitches_from_frame(pitches: pd.DataFrame) -> pd.DataFrame:
    pitches = pitches.copy()
    pitches["is_valid"], pitches["missing_fields"] = flag_missing_critical(pitches, CRITICAL_FIELDS)
    return pitches.sort_values(
        ["pitcher_id", "game_date", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)


def build_season_pitches(raw_path: Path) -> pd.DataFrame:
    return build_season_pitches_from_frame(build_pitch_frame(raw_path))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the per-pitch feature table from raw Statcast data.")
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
        logger.info("Building pitch table for %s", raw_path.name)
        season_df = build_season_pitches(raw_path)
        write_partitioned(season_df, args.output_dir)

    logger.info("Done. Wrote %d season(s) to %s", len(raw_files), args.output_dir)


if __name__ == "__main__":
    main()
