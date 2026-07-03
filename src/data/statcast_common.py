"""Shared loading, cleaning, and outcome-encoding logic for the processed Statcast tables.

Used by both `build_features.py` (per-pitch table) and `build_plate_appearances.py`
(per-plate-appearance table) so the two scripts share one definition of what a
"clean" pitch row looks like.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Maps a PA-ending `events` value to a single outcome label. Events left out here
# (catcher_interf, truncated_pa, and anything unrecognized) fall back to the
# description-based mapping below, since they don't cleanly fit one of these
# buckets and the description says what actually happened on the pitch itself.
EVENT_OUTCOME_MAP = {
    "strikeout": "strikeout",
    "strikeout_double_play": "strikeout",
    "walk": "walk",
    "intent_walk": "walk",
    "hit_by_pitch": "hit_by_pitch",
    "single": "single",
    "double": "double",
    "triple": "triple",
    "home_run": "home_run",
    "field_out": "hit_into_play_out",
    "force_out": "hit_into_play_out",
    "grounded_into_double_play": "hit_into_play_out",
    "double_play": "hit_into_play_out",
    "triple_play": "hit_into_play_out",
    "fielders_choice_out": "hit_into_play_out",
    "sac_fly": "hit_into_play_out",
    "sac_fly_double_play": "hit_into_play_out",
    "sac_bunt": "hit_into_play_out",
    # Batter reaches base but isn't credited with a hit. No bucket for this in the
    # requested label set, so it's grouped with other in-play resolutions.
    "field_error": "hit_into_play_out",
    "fielders_choice": "hit_into_play_out",
}

# Used for pitches that don't end a plate appearance, and as the fallback for
# events not present in EVENT_OUTCOME_MAP.
DESCRIPTION_OUTCOME_MAP = {
    "ball": "ball",
    "blocked_ball": "ball",
    "automatic_ball": "ball",
    "pitchout": "ball",
    "called_strike": "called_strike",
    "automatic_strike": "called_strike",
    "swinging_strike": "swinging_strike",
    "swinging_strike_blocked": "swinging_strike",
    "missed_bunt": "swinging_strike",
    "foul": "foul",
    "foul_tip": "foul",
    "foul_bunt": "foul",
    "bunt_foul_tip": "foul",
    "hit_by_pitch": "hit_by_pitch",
    "hit_into_play": "hit_into_play_out",
}


def compute_outcome(events: pd.Series, description: pd.Series) -> pd.Series:
    """Encode `events`/`description` into a single categorical outcome label.

    `events` (populated only on the pitch that ends a plate appearance) takes
    priority; pitches mid-PA, and any PA-ending event not in EVENT_OUTCOME_MAP,
    fall back to `description`. Anything still unmapped comes back as NaN so it
    gets flagged as a missing critical field rather than mislabeled.
    """
    outcome = events.map(EVENT_OUTCOME_MAP)
    fallback_mask = outcome.isna()
    outcome = outcome.where(~fallback_mask, description.map(DESCRIPTION_OUTCOME_MAP))
    return outcome


def best_spin_rate(raw: pd.DataFrame) -> pd.Series:
    """release_spin_rate is the modern Statcast column. spin_rate_deprecated was
    the pre-2015 equivalent; it's coalesced in defensively even though it's been
    empty in every season pulled by fetch_statcast.py so far (spin wasn't tracked
    at all before 2015)."""
    return raw["release_spin_rate"].combine_first(raw["spin_rate_deprecated"])


def discover_raw_seasons(raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    return sorted(raw_dir.glob("statcast_*.parquet"))


def load_raw_season(path: Path) -> pd.DataFrame:
    logger.info("Loading %s", path)
    return pd.read_parquet(path)


def build_pitch_frame_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Select, rename, and clean the columns needed for modeling from a raw
    Statcast DataFrame. One row per pitch. Does not sort or flag rows -- callers
    decide sort order and critical fields for their own output grain."""
    df = pd.DataFrame(
        {
            "pitcher_id": raw["pitcher"],
            "batter_id": raw["batter"],
            "game_date": pd.to_datetime(raw["game_date"]),
            "game_pk": raw["game_pk"],
            "at_bat_number": raw["at_bat_number"],
            "pitch_number": raw["pitch_number"],
            "pitch_type": raw["pitch_type"],
            "release_speed": raw["release_speed"],
            "spin_rate": best_spin_rate(raw),
            "plate_x": raw["plate_x"],
            "plate_z": raw["plate_z"],
            "balls": raw["balls"],
            "strikes": raw["strikes"],
            "outs_when_up": raw["outs_when_up"],
            "inning": raw["inning"],
            "stand": raw["stand"],
            "p_throws": raw["p_throws"],
            "home_team": raw["home_team"],
            "away_team": raw["away_team"],
            "season": raw["game_year"],
        }
    )
    df["outcome"] = compute_outcome(raw["events"], raw["description"])
    return df


def build_pitch_frame(raw_path: Path) -> pd.DataFrame:
    return build_pitch_frame_from_raw(load_raw_season(raw_path))


def flag_missing_critical(df: pd.DataFrame, critical_fields: list[str]) -> tuple[pd.Series, pd.Series]:
    """Return (is_valid, missing_fields) for `critical_fields` without dropping
    any rows, so callers can see exactly what's missing and decide whether to
    filter it out downstream."""
    flags = pd.DataFrame(
        {col: np.where(df[col].isna(), col + ",", "") for col in critical_fields},
        index=df.index,
    )
    missing_fields = flags.sum(axis=1).str.rstrip(",")
    is_valid = missing_fields == ""
    return is_valid, missing_fields


def write_partitioned(df: pd.DataFrame, output_dir: Path) -> None:
    n_flagged = int((~df["is_valid"]).sum())
    logger.info(
        "%d/%d rows flagged with missing critical fields (%.1f%%)",
        n_flagged,
        len(df),
        100 * n_flagged / len(df) if len(df) else 0.0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir, partition_cols=["season"], index=False)


def read_partitioned(dataset_dir: Path) -> pd.DataFrame:
    """Read a season-partitioned dataset written by `write_partitioned`.

    Plain `pd.read_parquet(dataset_dir)` raises `NotImplementedError` here:
    pyarrow's hive-partition discovery encodes the `season` partition column as
    `dictionary<values=int32>`, a type pandas' nullable-dtype conversion can't
    reconstruct. Decode any such dictionary columns back to their plain value
    type before handing off to pandas.
    """
    table = pq.read_table(dataset_dir)
    for i, field in enumerate(table.schema):
        if pa.types.is_dictionary(field.type):
            table = table.set_column(i, field.name, table.column(i).cast(field.type.value_type))
    return table.to_pandas()
