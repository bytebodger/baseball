"""Leak-safe, strictly-prior batted-ball-quality feature: a pitcher's
rolling average exit velocity, hard-hit rate, and BABIP *allowed*, and a
batter's rolling average exit velocity and hard-hit rate *produced*.

The raw Statcast files (data/raw/statcast_*.parquet) carry `launch_speed`,
`launch_angle`, and `estimated_ba_using_speedangle` for every batted ball --
but build_pitch_frame_from_raw (src/data/statcast_common.py) never selects
these columns when building the processed pitch table, so they've never
been available to any dataset or model in this project. This module reads
them directly from the raw files instead of the processed table.

Same "sorted per-player date array + searchsorted for a strictly-prior
cutoff" pattern PitcherWorkloadHistory/workload_features_for
(src/models/bullpen_availability.py) already use for per-pitcher rolling
features, applied here to both the pitcher (contact allowed) and batter
(contact produced) perspectives from the same underlying batted-ball event
stream. Deliberately an EXPANDING (career-to-date) average, not a fixed
trailing window: BABIP/hard-hit-rate-allowed are well known to need several
hundred batted-ball events to stabilize (a single season's ~450 batted
balls faced is already a small sample -- see this project's own probe
showing a 74-event real sample swinging BABIP by several points on pure
noise), so a short window would be dominated by noise a wider one damps
out. A player with fewer than MIN_BATTED_BALLS_FOR_STABLE_ESTIMATE prior
batted-ball events falls back to the league average for both features --
same "not enough signal yet" spirit as workload_features_for's 365-day
sentinel and LeagueRatesIndex's earliest-known-season fallback.

A batted ball is identified by the raw Statcast `type == "X"` flag (a ball
that ended the plate appearance in play), NOT by `launch_speed.notna()`
alone -- an earlier version of this module used the latter and was
silently counting foul balls as "contact allowed": Statcast tracks exit
velocity on fouls too (a real batted ball, just not a fair one), and fouls
turned out to be ~48% of the rows with a non-null launch_speed in a real
season's data. That's a real, meaningful contamination of the exit-velo/
hard-hit-rate-allowed features every checkpoint before this fix was built
from, not a rounding error.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.statcast_common import RAW_DATA_DIR, TRAIN_SEASON_RANGE, VAL_SEASONS, compute_outcome, discover_raw_seasons, load_raw_season
from src.training.pretrain_long_history_encoder import NS_PER_DAY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = Path("checkpoints") / "contact_quality.pkl"

# Standard Statcast "hard-hit" definition: exit velocity >= 95 mph.
HARD_HIT_THRESHOLD_MPH = 95.0

# A player with fewer prior batted-ball events than this falls back to the
# league average rather than a real (but noisy) small-sample estimate. Well
# below the ~800 balls in play sabermetric convention treats as fully
# stabilized -- the point here isn't a stabilized true-talent estimate, just
# enough real signal to be better than pure league-average, which stays the
# fallback below this threshold precisely because a smaller sample isn't
# reliable enough to trust outright.
MIN_BATTED_BALLS_FOR_STABLE_ESTIMATE = 50

PERSPECTIVES = ("pitcher", "batter")

# BABIP excludes home runs from both the numerator and denominator (a home
# run isn't "in play" for a fielder in any meaningful sense -- see standard
# sabermetric convention). "Hit" for BABIP purposes is single/double/triple
# only -- reuses statcast_common.compute_outcome's own event classification
# (the same one build_pitch_frame_from_raw uses for the processed pitch
# table's `outcome` column) rather than reimplementing event-name mapping
# here, so a batted ball's classification can never drift out of sync with
# how the rest of this project already labels it.
BABIP_HIT_OUTCOMES = {"single", "double", "triple"}


def load_raw_batted_balls(
    raw_dir: Path = RAW_DATA_DIR, season_start: int | None = None, season_end: int | None = None
) -> pd.DataFrame:
    """One row per real batted-ball event (`type == "X"` in the raw
    Statcast schema -- a ball that ended the plate appearance in play, NOT
    a foul; see module docstring), across every raw season file in
    `raw_dir`, restricted to [season_start, season_end] inclusive when
    given. Columns: pitcher_id, batter_id, game_date, launch_speed,
    hard_hit, outcome, is_home_run, is_babip_hit -- outcome/is_home_run/
    is_babip_hit renamed/derived from the raw events/description columns
    via the same compute_outcome this project's processed pitch table
    itself uses.
    """
    frames = []
    for path in discover_raw_seasons(raw_dir):
        raw = load_raw_season(path)
        raw = raw[raw["type"] == "X"]
        if len(raw) == 0:
            continue
        game_date = pd.to_datetime(raw["game_date"])
        season = game_date.dt.year
        if season_start is not None:
            raw = raw[season >= season_start]
            game_date = game_date[season >= season_start]
            season = season[season >= season_start]
        if season_end is not None:
            raw = raw[season <= season_end]
            game_date = game_date[season <= season_end]
        outcome = compute_outcome(raw["events"], raw["description"])
        frames.append(
            pd.DataFrame(
                {
                    "pitcher_id": raw["pitcher"].to_numpy(),
                    "batter_id": raw["batter"].to_numpy(),
                    "game_date": game_date.to_numpy(),
                    "launch_speed": raw["launch_speed"].to_numpy(dtype="float64"),
                    "outcome": outcome.to_numpy(),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["pitcher_id", "batter_id", "game_date", "launch_speed", "outcome", "hard_hit", "is_home_run", "is_babip_hit"])
    batted_balls = pd.concat(frames, ignore_index=True)
    batted_balls = batted_balls[batted_balls["launch_speed"].notna()].reset_index(drop=True)  # defensive; type=="X" should always have it
    batted_balls["hard_hit"] = (batted_balls["launch_speed"] >= HARD_HIT_THRESHOLD_MPH).astype("float64")
    batted_balls["is_home_run"] = batted_balls["outcome"] == "home_run"
    batted_balls["is_babip_hit"] = batted_balls["outcome"].isin(BABIP_HIT_OUTCOMES).astype("float64")
    return batted_balls


@dataclass
class ContactQualityHistory:
    dates_by_player: dict[int, np.ndarray]  # sorted int64 date_ns arrays
    exit_velo_by_player: dict[int, np.ndarray]  # parallel launch_speed values
    hard_hit_by_player: dict[int, np.ndarray]  # parallel 0.0/1.0 hard-hit flags
    league_avg_exit_velo: float
    league_avg_hard_hit_rate: float
    babip_dates_by_player: dict[int, np.ndarray]  # sorted int64 date_ns arrays, home-run rows EXCLUDED
    babip_hit_by_player: dict[int, np.ndarray]  # parallel 0.0/1.0 single/double/triple flags
    league_avg_babip: float


def build_contact_quality_history(batted_balls: pd.DataFrame, id_column: str) -> ContactQualityHistory:
    """`id_column`: "pitcher_id" for the contact-allowed perspective,
    "batter_id" for the contact-produced perspective -- same batted_balls
    table (see load_raw_batted_balls), grouped from whichever side's history
    is being built. `batted_balls` should already be restricted to whatever
    data the caller considers safe to use as the league-average fallback and
    per-player history (e.g. TRAIN_SEASON_RANGE + VAL_SEASONS only, so a
    TEST_SEASON_RANGE query's fallback and per-player arrays are never built
    from data a real query wouldn't have had -- see build_default_history).
    """
    dates_by_player: dict[int, np.ndarray] = {}
    exit_velo_by_player: dict[int, np.ndarray] = {}
    hard_hit_by_player: dict[int, np.ndarray] = {}
    for player_id, group in batted_balls.groupby(id_column):
        group = group.sort_values("game_date")
        dates_by_player[int(player_id)] = group["game_date"].to_numpy().astype("datetime64[ns]").astype("int64")
        exit_velo_by_player[int(player_id)] = group["launch_speed"].to_numpy(dtype="float64")
        hard_hit_by_player[int(player_id)] = group["hard_hit"].to_numpy(dtype="float64")

    league_avg_exit_velo = float(batted_balls["launch_speed"].mean()) if len(batted_balls) else 90.0
    league_avg_hard_hit_rate = float(batted_balls["hard_hit"].mean()) if len(batted_balls) else 0.35

    non_hr = batted_balls[~batted_balls["is_home_run"]]
    babip_dates_by_player: dict[int, np.ndarray] = {}
    babip_hit_by_player: dict[int, np.ndarray] = {}
    for player_id, group in non_hr.groupby(id_column):
        group = group.sort_values("game_date")
        babip_dates_by_player[int(player_id)] = group["game_date"].to_numpy().astype("datetime64[ns]").astype("int64")
        babip_hit_by_player[int(player_id)] = group["is_babip_hit"].to_numpy(dtype="float64")
    league_avg_babip = float(non_hr["is_babip_hit"].mean()) if len(non_hr) else 0.30

    return ContactQualityHistory(
        dates_by_player, exit_velo_by_player, hard_hit_by_player, league_avg_exit_velo, league_avg_hard_hit_rate,
        babip_dates_by_player, babip_hit_by_player, league_avg_babip,
    )


def contact_quality_features_for(
    history: ContactQualityHistory, player_id: int, cutoff_ns: int, min_events: int = MIN_BATTED_BALLS_FOR_STABLE_ESTIMATE
) -> tuple[float, float]:
    """(avg_exit_velo, hard_hit_rate) for `player_id` from every batted-ball
    event strictly before `cutoff_ns`. Falls back to the league average when
    the player has no history at all, or fewer than `min_events` prior
    events (see module docstring)."""
    dates = history.dates_by_player.get(player_id)
    if dates is None or len(dates) == 0:
        return history.league_avg_exit_velo, history.league_avg_hard_hit_rate

    end = int(np.searchsorted(dates, cutoff_ns, side="left"))
    if end == 0 or end < min_events:
        return history.league_avg_exit_velo, history.league_avg_hard_hit_rate

    exit_velos = history.exit_velo_by_player[player_id][:end]
    hard_hits = history.hard_hit_by_player[player_id][:end]
    return float(exit_velos.mean()), float(hard_hits.mean())


def babip_for(
    history: ContactQualityHistory, player_id: int, cutoff_ns: int, min_events: int = MIN_BATTED_BALLS_FOR_STABLE_ESTIMATE
) -> float:
    """BABIP for `player_id` from every non-home-run batted-ball event
    strictly before `cutoff_ns` (see module docstring for why home runs are
    excluded). Falls back to the league average the same way
    contact_quality_features_for does."""
    dates = history.babip_dates_by_player.get(player_id)
    if dates is None or len(dates) == 0:
        return history.league_avg_babip

    end = int(np.searchsorted(dates, cutoff_ns, side="left"))
    if end == 0 or end < min_events:
        return history.league_avg_babip

    hits = history.babip_hit_by_player[player_id][:end]
    return float(hits.mean())


def contact_quality_features_batch(
    history: ContactQualityHistory,
    player_ids: pd.Series,
    game_dates: pd.Series,
    min_events: int = MIN_BATTED_BALLS_FOR_STABLE_ESTIMATE,
) -> np.ndarray:
    """Vectorized-in-name-only convenience (still one lookup per row, each
    an O(1) dict access + a binary search over that player's own small
    date array) -- shape (n, 2): column 0 = avg_exit_velo, column 1 =
    hard_hit_rate. Same honest-naming convention as EmbeddingCache.get_batch."""
    cutoffs_ns = pd.to_datetime(game_dates).to_numpy().astype("datetime64[ns]").astype("int64")
    out = np.empty((len(player_ids), 2), dtype="float64")
    for i, (player_id, cutoff_ns) in enumerate(zip(player_ids, cutoffs_ns)):
        out[i, 0], out[i, 1] = contact_quality_features_for(history, int(player_id), int(cutoff_ns), min_events)
    return out


def babip_features_batch(
    history: ContactQualityHistory,
    player_ids: pd.Series,
    game_dates: pd.Series,
    min_events: int = MIN_BATTED_BALLS_FOR_STABLE_ESTIMATE,
) -> np.ndarray:
    """Vectorized-in-name-only convenience -- shape (n,): BABIP only (see
    babip_for). Split out from contact_quality_aux_targets_batch so a
    caller that's already computed contact_quality_features_batch for the
    same (player_ids, game_dates) -- e.g. EventDataset, which needs
    hard_hit_rate for both the input feature and the auxiliary target --
    doesn't redundantly re-run the same per-row exit-velo/hard-hit-rate
    lookup a second time just to reach its hard_hit_rate half."""
    cutoffs_ns = pd.to_datetime(game_dates).to_numpy().astype("datetime64[ns]").astype("int64")
    out = np.empty(len(player_ids), dtype="float64")
    for i, (player_id, cutoff_ns) in enumerate(zip(player_ids, cutoffs_ns)):
        out[i] = babip_for(history, int(player_id), int(cutoff_ns), min_events)
    return out


def contact_quality_aux_targets_batch(
    history: ContactQualityHistory,
    player_ids: pd.Series,
    game_dates: pd.Series,
    min_events: int = MIN_BATTED_BALLS_FOR_STABLE_ESTIMATE,
) -> np.ndarray:
    """The two real, leak-safe targets EventModel's auxiliary contact-
    quality head is trained to predict (see train_event_model.py) -- shape
    (n, 2): column 0 = BABIP, column 1 = hard_hit_rate, both "allowed" (or
    "produced") as of strictly before each row's own game_date. Standalone
    convenience for a caller that hasn't already computed
    contact_quality_features_batch for the same rows -- if it has (like
    EventDataset does, for the input feature), call babip_features_batch
    directly instead and reuse the hard_hit_rate column already in hand,
    rather than recomputing it here too."""
    cutoffs_ns = pd.to_datetime(game_dates).to_numpy().astype("datetime64[ns]").astype("int64")
    out = np.empty((len(player_ids), 2), dtype="float64")
    for i, (player_id, cutoff_ns) in enumerate(zip(player_ids, cutoffs_ns)):
        out[i, 0] = babip_for(history, int(player_id), int(cutoff_ns), min_events)
        _, out[i, 1] = contact_quality_features_for(history, int(player_id), int(cutoff_ns), min_events)
    return out


def save_contact_quality_histories(
    pitcher_history: ContactQualityHistory, batter_history: ContactQualityHistory, path: Path = DEFAULT_CHECKPOINT_PATH
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"pitcher": pitcher_history, "batter": batter_history}, f)


def load_contact_quality_histories(path: Path = DEFAULT_CHECKPOINT_PATH) -> dict[str, ContactQualityHistory]:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_default_histories(
    raw_dir: Path = RAW_DATA_DIR, season_start: int = TRAIN_SEASON_RANGE[0], season_end: int = VAL_SEASONS[-1]
) -> dict[str, ContactQualityHistory]:
    """The production build: restricted to [season_start, season_end], by
    default TRAIN_SEASON_RANGE + VAL_SEASONS (2015-2023) -- the same
    boundary event/hook/bullpen-availability models and (as of this
    session's earlier fix) BaserunningModel's rate table already use --
    TEST_SEASON_RANGE (2024-2025) is held out entirely, not just from model
    fitting, so a simulated TEST-season game's contact-quality lookup can
    never see that season's (or any other TEST-season game's) own real
    outcome baked into the history or the league-average fallback it falls
    back to. season_start/season_end are overridable for walk-forward
    retraining at a later season boundary, same reasoning as
    src/training/pretrain_encoder.py's load_season_split."""
    batted_balls = load_raw_batted_balls(raw_dir, season_start=season_start, season_end=season_end)
    logger.info("%d real batted-ball events loaded (seasons %d-%d)", len(batted_balls), season_start, season_end)
    pitcher_history = build_contact_quality_history(batted_balls, "pitcher_id")
    batter_history = build_contact_quality_history(batted_balls, "batter_id")
    logger.info(
        "Pitcher history: %d pitchers, league avg exit velo=%.1f mph, hard-hit rate=%.1f%%, BABIP=%.3f",
        len(pitcher_history.dates_by_player), pitcher_history.league_avg_exit_velo,
        pitcher_history.league_avg_hard_hit_rate * 100, pitcher_history.league_avg_babip,
    )
    logger.info(
        "Batter history: %d batters, league avg exit velo=%.1f mph, hard-hit rate=%.1f%%, BABIP=%.3f",
        len(batter_history.dates_by_player), batter_history.league_avg_exit_velo,
        batter_history.league_avg_hard_hit_rate * 100, batter_history.league_avg_babip,
    )
    return {"pitcher": pitcher_history, "batter": batter_history}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the leak-safe batted-ball-quality (exit velo / hard-hit rate / BABIP) history.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--train-season-start", type=int, default=TRAIN_SEASON_RANGE[0],
        help="Overrides the project-wide default train split start (statcast_common.TRAIN_SEASON_RANGE) -- "
        "e.g. for walk-forward retraining at a later season boundary.",
    )
    parser.add_argument(
        "--val-season-end", type=int, default=VAL_SEASONS[-1],
        help="Overrides the project-wide default validation split end (statcast_common.VAL_SEASONS[-1]) -- "
        "batted-ball events through this season (inclusive) are included.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    histories = build_default_histories(args.raw_dir, args.train_season_start, args.val_season_end)
    save_contact_quality_histories(histories["pitcher"], histories["batter"], args.checkpoint)
    logger.info("Saved contact-quality histories to %s", args.checkpoint)


if __name__ == "__main__":
    main()
