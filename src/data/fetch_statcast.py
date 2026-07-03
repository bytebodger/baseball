"""Download pitch-level Statcast data season by season into data/raw/.

Each season is pulled in monthly chunks (pybaseball's `statcast` truncates
results around 30,000 rows per query, which a single full-season request
can exceed) and concatenated into one Parquet file per season.
"""

import argparse
import logging
from calendar import monthrange
from datetime import date
from pathlib import Path

import pandas as pd
from pybaseball import statcast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# MLB regular/postseason runs roughly March (spring training) through November (World Series).
SEASON_START_MONTH = 3
SEASON_END_MONTH = 11


def month_chunks(year: int, today: date):
    """Yield (start_dt, end_dt) ISO date strings for each season month, capped at `today`."""
    for month in range(SEASON_START_MONTH, SEASON_END_MONTH + 1):
        start = date(year, month, 1)
        if start > today:
            break
        end = date(year, month, monthrange(year, month)[1])
        if end > today:
            end = today
        yield start.isoformat(), end.isoformat()


def fetch_season(year: int, raw_dir: Path = RAW_DATA_DIR, force: bool = False) -> None:
    """Download one season of Statcast data and save it as a Parquet file, unless cached."""
    out_path = raw_dir / f"statcast_{year}.parquet"
    if out_path.exists() and not force:
        logger.info("Season %d already downloaded, skipping (%s)", year, out_path)
        return

    today = date.today()
    chunks = []
    for start_dt, end_dt in month_chunks(year, today):
        logger.info("Fetching %d: %s to %s", year, start_dt, end_dt)
        df = statcast(start_dt=start_dt, end_dt=end_dt)
        if df is None or df.empty:
            logger.info("No rows for %s to %s", start_dt, end_dt)
            continue
        chunks.append(df)

    if not chunks:
        logger.warning("No data found for season %d, nothing written", year)
        return

    season_df = pd.concat(chunks, ignore_index=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    season_df.to_parquet(out_path, index=False)
    logger.info("Saved %d rows for season %d to %s", len(season_df), year, out_path)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pitch-level Statcast data season by season.")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--force", action="store_true", help="Re-download seasons even if a cached file exists.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    for year in range(args.start_year, args.end_year + 1):
        fetch_season(year, force=args.force)


if __name__ == "__main__":
    main()
