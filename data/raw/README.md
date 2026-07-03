# data/raw/

Untouched pitch-level Statcast data pulled directly from Baseball Savant via
`pybaseball`, produced by `src/data/fetch_statcast.py`.

- One file per season: `statcast_{year}.parquet`
- Each file is the concatenation of that season's monthly pull (March through
  November, capped at today's date for the current season)
- Files here are not tracked in git (see `.gitignore`) — regenerate with:

  ```bash
  python -m src.data.fetch_statcast --start-year 2010 --end-year 2026
  ```

  Pass `--force` to re-download a season even if its Parquet file already exists.

Do not hand-edit these files; any cleaning/feature engineering should read
from here and write to `data/processed/` instead.
