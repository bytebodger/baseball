"""Leak-safe, strictly-prior rolling XBH-rate-allowed history for pitchers --
built for the Phase 11 boundary-2 simulation-layer XBH calibration correction
(see src/simulation/xbh_calibration.py), not the event model's trunk.

Separate from src.data.contact_quality's ContactQualityHistory (exit velo /
hard-hit rate / BABIP) rather than an extension of it, deliberately: this
avoids touching or rebuilding the existing contact_quality.pkl checkpoint
(already baked into the trunk's raw-scalar feature and the aux head's real
target) for a mechanism that operates entirely at the simulation layer and
may not end up kept.

Same "sorted per-player date array + searchsorted for a strictly-prior
cutoff" pattern as ContactQualityHistory/babip_for, applied to the same
underlying real batted-ball event stream (src.data.contact_quality.
load_raw_batted_balls), just with XBH (double/triple/home_run) as the target
outcome instead of BABIP-eligible hits. A player with fewer than
MIN_BATTED_BALLS_FOR_STABLE_ESTIMATE prior batted-ball events falls back to
the league average, same convention as every other rolling stat in this
project.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.contact_quality import MIN_BATTED_BALLS_FOR_STABLE_ESTIMATE, load_raw_batted_balls
from src.data.statcast_common import RAW_DATA_DIR, TRAIN_SEASON_RANGE, VAL_SEASONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = Path("checkpoints") / "xbh_rate_history.pkl"

XBH_OUTCOMES = {"double", "triple", "home_run"}


@dataclass
class XbhRateHistory:
    dates_by_player: dict[int, np.ndarray]  # sorted int64 date_ns arrays
    is_xbh_by_player: dict[int, np.ndarray]  # parallel 0.0/1.0 XBH flags
    league_avg_xbh_rate: float


def build_xbh_rate_history(batted_balls: pd.DataFrame, id_column: str) -> XbhRateHistory:
    """`id_column`: "pitcher_id" for the allowed perspective (the only one
    this correction currently uses). `batted_balls` should already be
    restricted to whatever data the caller considers safe for both the
    league-average fallback and the per-player history -- see
    build_default_xbh_rate_history."""
    is_xbh = batted_balls["outcome"].isin(XBH_OUTCOMES).astype("float64")
    batted_balls = batted_balls.assign(is_xbh=is_xbh)

    dates_by_player: dict[int, np.ndarray] = {}
    is_xbh_by_player: dict[int, np.ndarray] = {}
    for player_id, group in batted_balls.groupby(id_column):
        group = group.sort_values("game_date")
        dates_by_player[int(player_id)] = group["game_date"].to_numpy().astype("datetime64[ns]").astype("int64")
        is_xbh_by_player[int(player_id)] = group["is_xbh"].to_numpy(dtype="float64")

    league_avg_xbh_rate = float(is_xbh.mean()) if len(batted_balls) else 0.06
    return XbhRateHistory(dates_by_player, is_xbh_by_player, league_avg_xbh_rate)


def xbh_rate_for(
    history: XbhRateHistory, player_id: int, cutoff_ns: int, min_events: int = MIN_BATTED_BALLS_FOR_STABLE_ESTIMATE
) -> float:
    """XBH rate (of batted balls) for `player_id` from every batted-ball
    event strictly before `cutoff_ns`. Falls back to the league average when
    the player has no history at all, or fewer than `min_events` prior
    events -- same convention as contact_quality.babip_for."""
    dates = history.dates_by_player.get(player_id)
    if dates is None or len(dates) == 0:
        return history.league_avg_xbh_rate

    end = int(np.searchsorted(dates, cutoff_ns, side="left"))
    if end == 0 or end < min_events:
        return history.league_avg_xbh_rate

    return float(history.is_xbh_by_player[player_id][:end].mean())


def xbh_rate_features_batch(
    history: XbhRateHistory, player_ids: pd.Series, game_dates: pd.Series, min_events: int = MIN_BATTED_BALLS_FOR_STABLE_ESTIMATE
) -> np.ndarray:
    """Vectorized-in-name-only convenience -- shape (n,), same pattern as
    contact_quality.babip_features_batch."""
    cutoffs_ns = pd.to_datetime(game_dates).to_numpy().astype("datetime64[ns]").astype("int64")
    out = np.empty(len(player_ids), dtype="float64")
    for i, (player_id, cutoff_ns) in enumerate(zip(player_ids, cutoffs_ns)):
        out[i] = xbh_rate_for(history, int(player_id), int(cutoff_ns), min_events)
    return out


def save_xbh_rate_history(history: XbhRateHistory, path: Path = DEFAULT_CHECKPOINT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(history, f)


def load_xbh_rate_history(path: Path = DEFAULT_CHECKPOINT_PATH) -> XbhRateHistory:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_default_xbh_rate_history(
    raw_dir: Path = RAW_DATA_DIR, season_start: int = TRAIN_SEASON_RANGE[0], season_end: int = VAL_SEASONS[-1]
) -> XbhRateHistory:
    """The production build: restricted to [season_start, season_end], by
    default TRAIN_SEASON_RANGE + VAL_SEASONS (2015-2023) -- same leak-safe
    boundary as contact_quality.build_default_histories -- TEST_SEASON_RANGE
    (2024-2025) is held out entirely, so a simulated TEST-season game's
    lookup can never see that season's own real outcome baked into the
    history or its league-average fallback."""
    batted_balls = load_raw_batted_balls(raw_dir, season_start=season_start, season_end=season_end)
    logger.info("%d real batted-ball events loaded (seasons %d-%d)", len(batted_balls), season_start, season_end)
    history = build_xbh_rate_history(batted_balls, "pitcher_id")
    logger.info(
        "Pitcher XBH-rate history: %d pitchers, league avg XBH rate=%.4f", len(history.dates_by_player), history.league_avg_xbh_rate
    )
    return history


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the leak-safe rolling pitcher XBH-rate-allowed history.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--train-season-start", type=int, default=TRAIN_SEASON_RANGE[0],
        help="Overrides the project-wide default train split start -- e.g. for walk-forward retraining at a later season boundary.",
    )
    parser.add_argument(
        "--val-season-end", type=int, default=VAL_SEASONS[-1],
        help="Overrides the project-wide default validation split end -- batted-ball events through this season (inclusive) are included.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    history = build_default_xbh_rate_history(args.raw_dir, args.train_season_start, args.val_season_end)
    save_xbh_rate_history(history, args.checkpoint)
    logger.info("Saved XBH-rate history to %s", args.checkpoint)


if __name__ == "__main__":
    main()
