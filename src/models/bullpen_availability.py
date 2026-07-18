"""Predicts the probability a reliever is available on a given day, from
their own recent workload (recent pitch counts, recent appearances, rest
days) -- not from a live 26-man roster or injury report, which don't exist
as a historical data source here (see game_dataset.py's own docstring on
the same limitation for its trailing-window bullpen proxy).

"Available" is never directly observed in this data: we only ever see
whether a pitcher *did* appear on a given day, not whether they *could*
have. This module uses "appeared in relief that day" as the closest
directly-observable proxy for "was available" -- the same proxy
game_dataset.py's `_bullpen_ids` already leans on implicitly (a trailing
`bullpen_window_days` of actual appearances stands in for roster
availability). A pitcher who was available but simply wasn't needed reads
here as a (slightly noisy) negative label; there's no way to distinguish
that case from "unavailable" using only Statcast's box-score-level
appearance record, and it's an honest limitation rather than one this
module tries to paper over.

Training examples: for every (team, game) in the historical record, the
candidate pool is exactly game_dataset.py's own bullpen definition --
pitchers who appeared for that team at all in the `window_days` strictly
before the game, excluding that game's own starter (see build_query_examples
and compare to GameOutcomeDataset._bullpen_ids). Each candidate is a
training row: label=1 if they actually relieved in that specific game,
label=0 if they didn't. This means validating against the same population
GameOutcomeDataset's bullpen embeddings are actually built from, not an
arbitrary redefinition of "candidate reliever."

Two candidate predictors are fit and compared honestly on held-out
(VAL_SEASONS) data -- a hand-set, not-fit-to-labels heuristic (rest days
help, heavy recent pitch counts hurt, back-to-back usage hurts) versus a
plain logistic regression fit on the same workload features. This becomes
the "general" model select_availability_model returns for every non-closer
candidate.

Beyond the aggregate AUC/accuracy/Brier comparison, this module also builds
a decile calibration table (see compute_calibration_table): are candidates
this predictor calls "70% available" actually available about 70% of the
time, not just ranked correctly relative to each other? AUC alone can't
answer that -- a model can discriminate well while still being badly
miscalibrated. That check is further split by bullpen role (closer/middle
reliever/long reliever, via classify_reliever_roles), and empirically
closers calibrate much worse under the general model than middle or long
relievers do (mean |calibration gap| roughly 3-5x theirs) -- the aggregate
number looks fine only because middle relievers dominate the population by
sheer count and mask it.

To fix that, select_availability_model also fits a dedicated closer path:
first a closer-only logistic regression (same WORKLOAD_FEATURE_NAMES, fit
on just the ~33 classified closers' own examples), and a middle-ground
RoleAwareLogisticRegressionModel -- one shared model fit across every role,
with a role-specific intercept and closer-specific coefficients on two
features built specifically for this (days_since_last_save_situation,
days_since_last_light_appearance -- see compute_entry_situations), pooling
statistical power across the full dataset rather than fitting closers'
comparatively tiny slice in total isolation. Whichever of the two actually
calibrates better on held-out closer examples (mean |calibration gap|) is
what ships for closers; the final BullpenAvailabilityPredictor dispatches
to it automatically for any candidate classify_reliever_roles calls a
"closer," and to the general model for everyone else.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.data.game_dataset import (
    BATTER_APPEARANCES_DIR,
    DEFAULT_BULLPEN_WINDOW_DAYS,
    GAMES_DIR,
    PITCHER_APPEARANCES_DIR,
    load_game_split,
)
from src.data.statcast_common import PROCESSED_DATA_DIR, RAW_DATA_DIR, TRAIN_SEASON_RANGE, VAL_SEASONS, read_partitioned
from src.training.pretrain_long_history_encoder import NS_PER_DAY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "bullpen_availability.yaml"
DEFAULT_CALIBRATION_PLOT_PATH = Path("logs") / "bullpen_availability_calibration.png"
DEFAULT_CHECKPOINT_PATH = Path("checkpoints") / "bullpen_availability.pkl"

# Trailing windows the workload features are computed over. 3 days covers
# "did they just throw a heavy outing," 7 days covers a fuller week's worth
# of usage -- both strictly shorter than window_days (the candidate-pool
# window), which only decides who's a candidate at all, not how fatigued
# they are.
SHORT_WINDOW_DAYS = 3
LONG_WINDOW_DAYS = 7

WORKLOAD_FEATURE_NAMES = [
    "days_since_last_appearance",
    "pitches_last_appearance",
    "pitches_trailing_short",
    "pitches_trailing_long",
    "appearances_trailing_long",
    "back_to_back",
]

# The two extra features RoleAwareLogisticRegressionModel gives closers
# their own coefficients on (see module docstring). "Light" and "save
# situation" are themselves heuristic proxies -- see LIGHT_APPEARANCE_MAX_PITCHES
# and the SAVE_SITUATION_* thresholds below for what they actually mean.
CLOSER_RECENCY_FEATURE_NAMES = ["days_since_last_save_situation", "days_since_last_light_appearance"]

# A "light" appearance: roughly a clean single inning, not a taxing stint --
# the kind of outing that lets a reliever (especially a closer, protected
# more carefully than the rest of the pen) come back again soon.
LIGHT_APPEARANCE_MAX_PITCHES = 20

# Approximates the real MLB save rule (entering with the tying run at least
# on deck) as "entered in the 9th inning or later with your team leading by
# 1-3 runs" -- close enough without the base/out state this pipeline
# doesn't carry, and the standard simplification for this kind of proxy.
SAVE_SITUATION_MIN_ENTRY_INNING = 9
SAVE_SITUATION_MIN_LEAD = 1
SAVE_SITUATION_MAX_LEAD = 3

# The closer-only model's own extra feature -- see the "Team-level
# save-opportunity history" section below for what it means and why it's
# closer-only rather than shared with the general/role-aware models.
TEAM_SAVE_OPPORTUNITY_FEATURE_NAME = "team_save_situations_trailing"
TEAM_SAVE_OPPORTUNITY_TRAILING_DAYS = LONG_WINDOW_DAYS
CLOSER_ONLY_FEATURE_NAMES = WORKLOAD_FEATURE_NAMES + [TEAM_SAVE_OPPORTUNITY_FEATURE_NAME]


@dataclass
class BullpenAvailabilityConfig:
    window_days: int = DEFAULT_BULLPEN_WINDOW_DAYS

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "BullpenAvailabilityConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


# ---------------------------------------------------------------------------
# Workload history: per-pitcher chronological appearance timeline with pitch
# counts (plus save-situation/light-appearance subsets for the closer path),
# supporting "strictly before cutoff" feature lookups the same way
# PlayerPitchSequenceDataset/build_chunk_index do for pitch-level history.
# ---------------------------------------------------------------------------


def compute_pitch_counts(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, pitcher_id): how many pitches they threw in
    that game. Statcast has no separate per-appearance pitch-count field --
    this is just a count of that pitcher's rows within the game."""
    return pitches.groupby(["game_pk", "pitcher_id"], as_index=False).size().rename(columns={"size": "pitches"})


def compute_entry_situations(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, pitcher_id): the inning they entered (their
    first pitch of that appearance) and whether that entry was a save
    situation -- entered in the 9th or later with their own (fielding) team
    leading by 1-3 runs, using inning_topbot to work out which side is
    which (see src/data/event_dataset.py's score_diff for the same
    inning_topbot-based convention). Also used by classify_reliever_roles
    for entry_inning alone (compute_entry_innings is a thin wrapper kept
    for that narrower, score-independent use).
    """
    sorted_pitches = pitches.sort_values(["game_pk", "pitcher_id", "inning", "at_bat_number", "pitch_number"])
    first_rows = sorted_pitches.drop_duplicates(subset=["game_pk", "pitcher_id"], keep="first")

    is_away_batting = first_rows["inning_topbot"] == "Top"
    fielding_score = first_rows["home_score"].where(is_away_batting, first_rows["away_score"])
    batting_score = first_rows["away_score"].where(is_away_batting, first_rows["home_score"])
    lead = (fielding_score - batting_score).to_numpy(dtype="float64")

    result = first_rows[["game_pk", "pitcher_id", "inning"]].rename(columns={"inning": "entry_inning"}).reset_index(drop=True)
    result["is_save_situation"] = (
        (first_rows["inning"].to_numpy() >= SAVE_SITUATION_MIN_ENTRY_INNING)
        & (lead >= SAVE_SITUATION_MIN_LEAD)
        & (lead <= SAVE_SITUATION_MAX_LEAD)
    )
    return result


def compute_entry_innings(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, pitcher_id): the first inning that pitcher
    threw a pitch in that game -- the signal role classification uses
    (closers enter late, long relievers/mop-up men often enter early).
    A thin, score-independent wrapper around compute_entry_situations."""
    return compute_entry_situations(pitches)[["game_pk", "pitcher_id", "entry_inning"]]


@dataclass
class PitcherWorkloadHistory:
    dates_by_pitcher: dict[int, np.ndarray]  # sorted int64 date_ns arrays
    pitches_by_pitcher: dict[int, np.ndarray]  # parallel pitch-count arrays
    save_dates_by_pitcher: dict[int, np.ndarray]  # sorted subset of dates that were save situations
    light_dates_by_pitcher: dict[int, np.ndarray]  # sorted subset of dates that were light (<=20 pitch) outings


def build_workload_history(
    pitcher_appearances: pd.DataFrame,
    pitch_counts: pd.DataFrame,
    entry_situations: pd.DataFrame | None = None,
) -> PitcherWorkloadHistory:
    """pitcher_appearances: game_pk/team/pitcher_id/game_date/season/is_starter
    (see game_dataset.py). Appearances with no matching pitch_counts row
    (shouldn't happen for real data -- every appearance threw at least one
    pitch -- but guarded rather than assumed) get 0 pitches rather than
    dropping the appearance itself, since the appearance/date is still real
    workload-history signal even if the pitch count is missing.

    entry_situations (see compute_entry_situations) is optional: pass it to
    also populate save_dates_by_pitcher/light_dates_by_pitcher for the
    closer path's recency features. Without it those two are just empty
    dicts, which closer_recency_features_for's sentinel handles the same
    way as an unknown pitcher.
    """
    merged = pitcher_appearances.merge(pitch_counts, on=["game_pk", "pitcher_id"], how="left")
    merged["pitches"] = merged["pitches"].fillna(0)
    if entry_situations is not None:
        merged = merged.merge(entry_situations[["game_pk", "pitcher_id", "is_save_situation"]], on=["game_pk", "pitcher_id"], how="left")
        # The left merge upcasts this bool column to object (NaN can't fit
        # in a bool column) for any unmatched row -- cast back explicitly,
        # or the boolean-array indexing below raises.
        merged["is_save_situation"] = merged["is_save_situation"].fillna(False).astype(bool)
    else:
        merged["is_save_situation"] = False
    merged["is_light"] = merged["pitches"] <= LIGHT_APPEARANCE_MAX_PITCHES
    merged = merged.sort_values(["pitcher_id", "game_date"])

    dates_by_pitcher: dict[int, np.ndarray] = {}
    pitches_by_pitcher: dict[int, np.ndarray] = {}
    save_dates_by_pitcher: dict[int, np.ndarray] = {}
    light_dates_by_pitcher: dict[int, np.ndarray] = {}
    for pitcher_id, group in merged.groupby("pitcher_id"):
        dates_ns = group["game_date"].to_numpy().astype("datetime64[ns]").astype("int64")
        dates_by_pitcher[pitcher_id] = dates_ns
        pitches_by_pitcher[pitcher_id] = group["pitches"].to_numpy(dtype="float64")
        save_dates_by_pitcher[pitcher_id] = dates_ns[group["is_save_situation"].to_numpy()]
        light_dates_by_pitcher[pitcher_id] = dates_ns[group["is_light"].to_numpy()]
    return PitcherWorkloadHistory(dates_by_pitcher, pitches_by_pitcher, save_dates_by_pitcher, light_dates_by_pitcher)


def workload_features_for(history: PitcherWorkloadHistory, pitcher_id: int, cutoff_ns: int) -> np.ndarray:
    """Workload features for `pitcher_id` as of strictly before `cutoff_ns`,
    in WORKLOAD_FEATURE_NAMES order. A pitcher with no appearances at all
    before the cutoff (shouldn't occur for a real candidate -- see
    build_query_examples, candidates are drawn from pitchers who *did*
    appear in the trailing window -- but guarded here too) gets a large
    days_since_last_appearance sentinel and zero for everything else."""
    dates = history.dates_by_pitcher.get(pitcher_id)
    if dates is None or len(dates) == 0:
        return np.array([365.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    end = int(np.searchsorted(dates, cutoff_ns, side="left"))
    if end == 0:
        return np.array([365.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    pitches = history.pitches_by_pitcher[pitcher_id]
    prior_dates = dates[:end]
    prior_pitches = pitches[:end]

    days_since_last = (cutoff_ns - prior_dates[-1]) / NS_PER_DAY
    pitches_last = prior_pitches[-1]

    short_start_ns = cutoff_ns - SHORT_WINDOW_DAYS * NS_PER_DAY
    long_start_ns = cutoff_ns - LONG_WINDOW_DAYS * NS_PER_DAY
    short_mask = prior_dates >= short_start_ns
    long_mask = prior_dates >= long_start_ns

    pitches_trailing_short = prior_pitches[short_mask].sum()
    pitches_trailing_long = prior_pitches[long_mask].sum()
    appearances_trailing_long = int(long_mask.sum())

    # "Back-to-back": appeared on each of the two calendar days immediately
    # preceding the cutoff -- a sharper fatigue signal than a trailing sum,
    # since two outings 6 days apart within the same trailing_long window
    # carry very different fatigue than two outings on consecutive days.
    one_day_ns = NS_PER_DAY
    appeared_yesterday = np.any((prior_dates >= cutoff_ns - one_day_ns) & (prior_dates < cutoff_ns))
    appeared_day_before = np.any((prior_dates >= cutoff_ns - 2 * one_day_ns) & (prior_dates < cutoff_ns - one_day_ns))
    back_to_back = float(appeared_yesterday and appeared_day_before)

    return np.array(
        [days_since_last, pitches_last, pitches_trailing_short, pitches_trailing_long, appearances_trailing_long, back_to_back]
    )


def _days_since_nearest_prior(dates: np.ndarray | None, cutoff_ns: int) -> float:
    if dates is None or len(dates) == 0:
        return 365.0
    end = int(np.searchsorted(dates, cutoff_ns, side="left"))
    if end == 0:
        return 365.0
    return float((cutoff_ns - dates[end - 1]) / NS_PER_DAY)


def closer_recency_features_for(history: PitcherWorkloadHistory, pitcher_id: int, cutoff_ns: int) -> tuple[float, float]:
    """(days_since_last_save_situation, days_since_last_light_appearance)
    for `pitcher_id` as of strictly before `cutoff_ns` -- the two features
    RoleAwareLogisticRegressionModel gives closers their own coefficients
    on. Same 365.0 sentinel convention as workload_features_for for a
    pitcher with no qualifying appearance at all before the cutoff."""
    save_dates = history.save_dates_by_pitcher.get(pitcher_id)
    light_dates = history.light_dates_by_pitcher.get(pitcher_id)
    return _days_since_nearest_prior(save_dates, cutoff_ns), _days_since_nearest_prior(light_dates, cutoff_ns)


# ---------------------------------------------------------------------------
# Team-level save-opportunity history: a closer-only-model-specific feature.
# How many save situations (see compute_entry_situations) has this TEAM's
# whole pitching staff faced recently -- distinct from
# closer_recency_features_for's days_since_last_save_situation, which is
# about that one pitcher's own last save chance. A team that's been in a lot
# of close, late games lately has been leaning on its closer (and its whole
# high-leverage relief mix) more than one whose games have mostly been
# decided early -- workload/fatigue signal the per-pitcher pitch-count
# features don't capture directly, since a closer can face a save situation
# and only throw 10 pitches, yet still be the busiest arm on the roster that
# week in terms of how often he's needed.
# ---------------------------------------------------------------------------
# TEAM_SAVE_OPPORTUNITY_FEATURE_NAME / TEAM_SAVE_OPPORTUNITY_TRAILING_DAYS /
# CLOSER_ONLY_FEATURE_NAMES are defined near the top of the file (with
# WORKLOAD_FEATURE_NAMES and friends), since CLOSER_ONLY_FEATURE_NAMES needs
# to exist before fit_logistic_regression_model's default argument does.


@dataclass
class TeamSaveOpportunityHistory:
    dates_by_team: dict[str, np.ndarray]  # sorted int64 date_ns arrays, one row per (team, game)
    save_situations_by_team: dict[str, np.ndarray]  # parallel counts: how many save situations that game


def compute_team_save_opportunity_counts(pitcher_appearances: pd.DataFrame, entry_situations: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game_pk): game_date and how many save-situation
    entries (almost always 0 or 1 -- the closer's own save chance, though a
    setup man entering in one too would count separately) that team's
    pitching staff faced in that game. Joins entry_situations (computed
    straight from `pitches`, no team column) onto pitcher_appearances
    (which already has the correct team per pitcher-game) rather than
    re-deriving home/away-team-to-fielding-team logic a second time."""
    merged = pitcher_appearances.merge(
        entry_situations[["game_pk", "pitcher_id", "is_save_situation"]], on=["game_pk", "pitcher_id"], how="left"
    )
    merged["is_save_situation"] = merged["is_save_situation"].fillna(False).astype(bool)
    return (
        merged.groupby(["team", "game_pk"], as_index=False)
        .agg(game_date=("game_date", "first"), save_situations=("is_save_situation", "sum"))
    )


def build_team_save_opportunity_history(team_save_counts: pd.DataFrame) -> TeamSaveOpportunityHistory:
    dates_by_team: dict[str, np.ndarray] = {}
    save_situations_by_team: dict[str, np.ndarray] = {}
    for team, group in team_save_counts.sort_values(["team", "game_date"]).groupby("team"):
        dates_by_team[team] = group["game_date"].to_numpy().astype("datetime64[ns]").astype("int64")
        save_situations_by_team[team] = group["save_situations"].to_numpy(dtype="float64")
    return TeamSaveOpportunityHistory(dates_by_team, save_situations_by_team)


def team_save_opportunities_trailing_for(
    history: TeamSaveOpportunityHistory, team: str, cutoff_ns: int, trailing_days: int = TEAM_SAVE_OPPORTUNITY_TRAILING_DAYS
) -> float:
    """Total save situations `team`'s staff faced in the `trailing_days`
    strictly before `cutoff_ns`. 0.0 (not a sentinel -- a team simply
    hasn't faced one) for an unknown team or one with no games that recently."""
    dates = history.dates_by_team.get(team)
    if dates is None or len(dates) == 0:
        return 0.0
    end = int(np.searchsorted(dates, cutoff_ns, side="left"))
    if end == 0:
        return 0.0
    start_ns = cutoff_ns - trailing_days * NS_PER_DAY
    prior_dates = dates[:end]
    prior_counts = history.save_situations_by_team[team][:end]
    return float(prior_counts[prior_dates >= start_ns].sum())


def attach_team_save_opportunity_feature(examples: pd.DataFrame, history: TeamSaveOpportunityHistory) -> pd.DataFrame:
    """Adds the TEAM_SAVE_OPPORTUNITY_FEATURE_NAME column -- `examples` must
    have "team" and "game_date" columns (i.e. be build_query_examples'
    output, or share its shape)."""
    examples = examples.copy()
    values = np.empty(len(examples))
    for i, (team, game_date) in enumerate(zip(examples["team"], examples["game_date"])):
        values[i] = team_save_opportunities_trailing_for(history, team, pd.Timestamp(game_date).value)
    examples[TEAM_SAVE_OPPORTUNITY_FEATURE_NAME] = values
    return examples


# ---------------------------------------------------------------------------
# Training examples: candidate pool + labels, mirroring
# GameOutcomeDataset._bullpen_ids exactly.
# ---------------------------------------------------------------------------


def build_query_examples(
    pitcher_appearances: pd.DataFrame, history: PitcherWorkloadHistory, window_days: int = DEFAULT_BULLPEN_WINDOW_DAYS
) -> pd.DataFrame:
    """One row per (pitcher_id, team, game_pk) candidate query: workload
    features as of strictly before that game's date, and label=1 if that
    candidate actually appeared in relief in that exact game.

    Candidates for a (team, game) are every pitcher who appeared for that
    team at all in the `window_days` strictly before the game, excluding
    that game's own starter -- identical to GameOutcomeDataset._bullpen_ids'
    own definition of a team's available bullpen, so this validates the
    real population that proxy is built from.
    """
    window_ns = window_days * NS_PER_DAY
    rows = []

    for team, team_group in pitcher_appearances.groupby("team"):
        team_group = team_group.sort_values("game_date")
        dates_ns = team_group["game_date"].to_numpy().astype("datetime64[ns]").astype("int64")
        pitcher_ids = team_group["pitcher_id"].to_numpy()

        for game_pk, game_group in team_group.groupby("game_pk"):
            cutoff = pd.Timestamp(game_group["game_date"].iloc[0])
            cutoff_ns = cutoff.value
            season = int(game_group["season"].iloc[0])

            starter_rows = game_group[game_group["is_starter"]]
            starter_id = starter_rows["pitcher_id"].iloc[0] if len(starter_rows) else None
            actual_relievers = set(game_group.loc[~game_group["is_starter"], "pitcher_id"])

            window_mask = (dates_ns >= cutoff_ns - window_ns) & (dates_ns < cutoff_ns)
            candidates = set(pitcher_ids[window_mask]) - {starter_id}

            for pid in candidates:
                features = workload_features_for(history, pid, cutoff_ns)
                rows.append(
                    [pid, team, game_pk, cutoff, season, int(pid in actual_relievers), *features]
                )

    columns = ["pitcher_id", "team", "game_pk", "game_date", "season", "label", *WORKLOAD_FEATURE_NAMES]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Bullpen-role classification and the role/closer-recency columns
# select_availability_model's closer path needs on top of build_query_examples'
# base output.
# ---------------------------------------------------------------------------

# Role thresholds are a heuristic proxy from usage patterns -- there's no
# official bullpen-role label anywhere in this data (Statcast doesn't carry
# one, and no separate roster/depth-chart source is wired into this
# project), same "closest available proxy, honestly labeled" spirit as
# game_dataset.py's own bullpen-availability window.
CLOSER_MIN_AVG_ENTRY_INNING = 8.5
CLOSER_MAX_AVG_PITCHES = 25.0
LONG_RELIEVER_MIN_AVG_PITCHES = 30.0
LONG_RELIEVER_MAX_AVG_ENTRY_INNING = 4.0
MIN_APPEARANCES_FOR_ROLE = 10


def classify_reliever_roles(
    pitcher_appearances: pd.DataFrame,
    entry_innings: pd.DataFrame,
    pitch_counts: pd.DataFrame,
    min_appearances: int = MIN_APPEARANCES_FOR_ROLE,
) -> pd.Series:
    """pitcher_id -> one of "closer"/"middle_reliever"/"long_reliever".
    Classified from a pitcher's *entire* relief-appearance history (not
    split by train/val season): this is a descriptive bucket for a
    validation report, not a feature fed into the general model, so there's
    no leakage concern in using their full record. (It IS used to build
    RoleAwareLogisticRegressionModel's training design matrix -- see that
    class's docstring for why that's still fine: the role bucket is a
    fixed, coarse usage-pattern label, not itself derived from any
    particular game's outcome.)

    - closer: overwhelmingly enters in the 9th inning (or later) for short
      outings -- avg entry inning >= 8.5 and avg pitches <= 25.
    - long_reliever: multi-inning stints, or entering early to mop up for a
      knocked-out starter -- avg pitches >= 30 or avg entry inning <= 4.
    - middle_reliever: everything else, the default bulk of a bullpen.

    Pitchers with fewer than `min_appearances` relief appearances aren't
    classified at all (too few outings for the average to mean anything)
    and are simply absent from the returned Series.
    """
    relief = pitcher_appearances[~pitcher_appearances["is_starter"]]
    relief = relief.merge(entry_innings, on=["game_pk", "pitcher_id"], how="left")
    relief = relief.merge(pitch_counts, on=["game_pk", "pitcher_id"], how="left")
    relief["pitches"] = relief["pitches"].fillna(0)

    agg = relief.groupby("pitcher_id").agg(
        n_appearances=("game_pk", "size"),
        avg_entry_inning=("entry_inning", "mean"),
        avg_pitches=("pitches", "mean"),
    )
    agg = agg[agg["n_appearances"] >= min_appearances]

    def _role(row) -> str:
        if row["avg_entry_inning"] >= CLOSER_MIN_AVG_ENTRY_INNING and row["avg_pitches"] <= CLOSER_MAX_AVG_PITCHES:
            return "closer"
        if row["avg_pitches"] >= LONG_RELIEVER_MIN_AVG_PITCHES or row["avg_entry_inning"] <= LONG_RELIEVER_MAX_AVG_ENTRY_INNING:
            return "long_reliever"
        return "middle_reliever"

    return agg.apply(_role, axis=1)


def attach_roles(examples: pd.DataFrame, roles: pd.Series) -> pd.DataFrame:
    """Adds a "role" column (closer/middle_reliever/long_reliever) via
    `roles` (pitcher_id -> role, see classify_reliever_roles); pitchers
    below the classification's min_appearances threshold map to
    "unclassified" rather than NaN, so downstream role comparisons
    (`role == "closer"`) don't need a separate null check."""
    examples = examples.copy()
    examples["role"] = examples["pitcher_id"].map(roles).fillna("unclassified")
    return examples


def attach_closer_recency_features(examples: pd.DataFrame, history: PitcherWorkloadHistory) -> pd.DataFrame:
    """Adds the CLOSER_RECENCY_FEATURE_NAMES columns (days since the
    pitcher's last save-situation appearance / last light appearance),
    computed the same "strictly before cutoff" way as workload_features_for,
    just against the two extra date arrays build_workload_history populates
    when given entry_situations. `examples` must have "pitcher_id" and
    "game_date" columns (i.e. be build_query_examples' output, or share its
    shape)."""
    examples = examples.copy()
    save_recency = np.empty(len(examples))
    light_recency = np.empty(len(examples))
    for i, (pitcher_id, game_date) in enumerate(zip(examples["pitcher_id"], examples["game_date"])):
        cutoff_ns = pd.Timestamp(game_date).value
        save_recency[i], light_recency[i] = closer_recency_features_for(history, pitcher_id, cutoff_ns)
    examples[CLOSER_RECENCY_FEATURE_NAMES[0]] = save_recency
    examples[CLOSER_RECENCY_FEATURE_NAMES[1]] = light_recency
    return examples


def compute_calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Deciles (or n_bins quantile bins) of predicted probability: for each
    bin, its sample count, mean predicted probability, and actual observed
    appearance rate -- a reliability curve as a directly inspectable table,
    not only a plot. Falls back to fewer bins (pd.qcut's duplicates="drop")
    when there are fewer than n_bins distinct predicted-probability values
    to split on -- unavoidable with a modest, mostly-integer feature set,
    which produces many repeated predicted probabilities."""
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["bin"] = pd.qcut(df["y_prob"], q=n_bins, duplicates="drop")
    table = (
        df.groupby("bin", observed=True)
        .agg(n=("y_true", "size"), mean_predicted=("y_prob", "mean"), observed_rate=("y_true", "mean"))
        .reset_index(drop=True)
    )
    table["calibration_gap"] = table["mean_predicted"] - table["observed_rate"]
    return table


def mean_abs_calibration_gap(table: pd.DataFrame) -> float:
    return float(table["calibration_gap"].abs().mean())


def calibration_by_role(
    examples: pd.DataFrame, y_prob: np.ndarray, roles: pd.Series, n_bins: int = 10
) -> dict[str, pd.DataFrame]:
    """examples: a build_query_examples-shaped DataFrame (must include a
    "pitcher_id" column, same row order as y_prob). Returns {role:
    calibration_table} for "overall" (every row, regardless of role) plus
    every role with at least n_bins examples -- fewer than that and decile
    bins stop being meaningful, so that role is skipped (logged, not
    silently dropped) rather than reported on a handful of points."""
    role_series = examples["pitcher_id"].map(roles)
    tables = {"overall": compute_calibration_table(examples["label"].to_numpy(), y_prob, n_bins)}
    for role in sorted(role_series.dropna().unique()):
        mask = (role_series == role).to_numpy()
        if mask.sum() < n_bins:
            logger.warning("Skipping calibration-by-role for %r: only %d examples (need >= %d)", role, mask.sum(), n_bins)
            continue
        tables[role] = compute_calibration_table(examples.loc[mask, "label"].to_numpy(), y_prob[mask], n_bins)
    return tables


def plot_calibration_reliability(tables: dict[str, pd.DataFrame], output_path: Path) -> None:
    """tables: {name: calibration_table} as returned by calibration_by_role
    -- one reliability curve per name, overlaid, same style as
    backtest.py's plot_calibration for GamePredictor."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    for name, table in tables.items():
        ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", label=name)

    ax.set_xlabel("Mean predicted availability probability (decile bin)")
    ax.set_ylabel("Observed appearance rate")
    ax.set_title("Bullpen availability calibration")
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Candidate predictors.
# ---------------------------------------------------------------------------


class HeuristicAvailabilityModel:
    """Fixed, hand-reasoned weights -- never fit to the appearance labels --
    on standardized workload features: more rest raises the score, a heavy
    last outing and back-to-back usage lower it. This is the "well-justified
    heuristic" baseline a trained model has to actually beat, not a strawman.
    """

    # (feature_name, weight, clip_max). Positive weight = raises availability.
    _TERMS = [
        ("days_since_last_appearance", 1.0, 3.0),  # more rest, up to a 3-day cap, helps
        ("pitches_last_appearance", -1.0, 50.0),  # a heavy last outing hurts
        ("back_to_back", -1.8, 1.0),  # already worked 2 straight days hurts a lot
    ]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = np.zeros(len(X))
        for name, weight, clip_max in self._TERMS:
            normalized = np.clip(X[name].to_numpy(dtype="float64"), 0.0, clip_max) / clip_max
            score += weight * normalized
        return 1.0 / (1.0 + np.exp(-2.5 * score))


def fit_logistic_regression_model(
    train_examples: pd.DataFrame, feature_names: list[str] = WORKLOAD_FEATURE_NAMES
) -> LogisticRegression:
    """feature_names defaults to WORKLOAD_FEATURE_NAMES (the general and
    plain closer-only fits); pass CLOSER_ONLY_FEATURE_NAMES to also include
    TEAM_SAVE_OPPORTUNITY_FEATURE_NAME, which `examples` must then already
    carry (see attach_team_save_opportunity_feature)."""
    model = LogisticRegression(max_iter=1000)
    model.fit(train_examples[feature_names].to_numpy(), train_examples["label"].to_numpy())
    return model


class RoleAwareLogisticRegressionModel:
    """A single shared logistic regression fit across every role, with a
    role-specific intercept (dummy columns for closer/long_reliever --
    middle relievers and unclassified pitchers are the implicit baseline)
    and closer-specific coefficients on the two features hypothesized to
    matter most for closer usage (see module docstring): recency of
    save-situation appearances and recency of light (low-pitch-count)
    appearances, via is_closer interaction terms. Every other workload
    feature keeps one shared coefficient across all roles, pooling
    statistical power across the full training set rather than fitting a
    role's own (possibly tiny) slice in isolation -- the middle ground
    between "one model for everyone" and "a separate model per role."

    For non-closer rows the two interaction columns are structurally zero
    (is_closer=0), so save/light recency has no effect on non-closer
    predictions at all -- only closers' predictions actually depend on them.
    `examples` must carry a "role" column (see attach_roles) and the
    CLOSER_RECENCY_FEATURE_NAMES columns (see attach_closer_recency_features)
    in addition to WORKLOAD_FEATURE_NAMES.
    """

    def __init__(self) -> None:
        self.logistic = LogisticRegression(max_iter=1000)

    def _design_matrix(self, examples: pd.DataFrame) -> np.ndarray:
        base = examples[WORKLOAD_FEATURE_NAMES].to_numpy(dtype="float64")
        is_closer = (examples["role"] == "closer").to_numpy(dtype="float64")
        is_long = (examples["role"] == "long_reliever").to_numpy(dtype="float64")
        interactions = examples[CLOSER_RECENCY_FEATURE_NAMES].to_numpy(dtype="float64") * is_closer[:, None]
        return np.column_stack([base, is_closer, is_long, interactions])

    def fit(self, examples: pd.DataFrame) -> "RoleAwareLogisticRegressionModel":
        self.logistic.fit(self._design_matrix(examples), examples["label"].to_numpy())
        return self

    def predict_proba(self, examples: pd.DataFrame) -> np.ndarray:
        return self.logistic.predict_proba(self._design_matrix(examples))[:, 1]


def _predict_proba(model, examples: pd.DataFrame) -> np.ndarray:
    if isinstance(model, HeuristicAvailabilityModel):
        return model.predict_proba(examples)
    return model.predict_proba(examples[WORKLOAD_FEATURE_NAMES].to_numpy())[:, 1]


def _closer_predict_proba(kind: str, model, examples: pd.DataFrame) -> np.ndarray:
    if kind == "role_aware_logistic_regression":
        return model.predict_proba(examples)
    if kind == "closer_only_logistic_regression":
        return model.predict_proba(examples[CLOSER_ONLY_FEATURE_NAMES].to_numpy())[:, 1]
    raise ValueError(f"unknown closer_kind: {kind!r}")


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Direct, honest validation: overall discrimination (AUC, accuracy,
    Brier) plus the exact breakdown requested -- mean predicted probability
    and the @0.5 hit rate separately for candidates who actually appeared
    (label=1) versus those who didn't (label=0)."""
    predicted_available = y_prob >= 0.5
    positive = y_true == 1
    negative = ~positive
    return {
        "n": len(y_true),
        "n_appeared": int(positive.sum()),
        "n_did_not_appear": int(negative.sum()),
        "auc": float(roc_auc_score(y_true, y_prob)) if positive.any() and negative.any() else float("nan"),
        "accuracy": float((predicted_available == positive).mean()),
        "brier_score": float(np.mean((y_prob - y_true) ** 2)),
        # Of relievers who actually appeared, how often we said "available."
        "mean_prob_when_appeared": float(y_prob[positive].mean()) if positive.any() else float("nan"),
        "hit_rate_when_appeared": float(predicted_available[positive].mean()) if positive.any() else float("nan"),
        # Of candidates who didn't appear, how often we said "not available."
        "mean_prob_when_not_appeared": float(y_prob[negative].mean()) if negative.any() else float("nan"),
        "hit_rate_when_not_appeared": float((~predicted_available[negative]).mean()) if negative.any() else float("nan"),
    }


# ---------------------------------------------------------------------------
# Packaged predictor + persistence.
# ---------------------------------------------------------------------------


@dataclass
class BullpenAvailabilityPredictor:
    kind: str  # "heuristic" or "logistic_regression" -- the general-population model
    model: object
    config: BullpenAvailabilityConfig
    closer_kind: str | None = None  # "closer_only_logistic_regression" or "role_aware_logistic_regression"
    closer_model: object | None = None
    roles: dict[int, str] | None = None  # pitcher_id -> role, for dispatching to the closer path
    team_save_history: "TeamSaveOpportunityHistory | None" = None  # only needed when closer_kind is closer_only

    def __post_init__(self) -> None:
        # classify_reliever_roles returns a pd.Series (the right shape for
        # its own pandas-idiom callers -- attach_roles' .map(), calibration_
        # by_role's .map() -- both accept a dict just as well). Normalized
        # to a plain dict here, once, since this field is read via a scalar
        # .get(pitcher_id, ...) hundreds of times per simulated game in the
        # hot replacement-selection loop (batched_select_replacements),
        # where a pandas Series' per-call lookup overhead dominated once
        # baserunning's and park-factor's own equivalent lookups were fixed
        # -- see BaserunningModel.__post_init__ and LeagueRatesIndex for the
        # same pattern applied earlier.
        if isinstance(self.roles, pd.Series):
            self.roles = self.roles.to_dict()

    def _role_for(self, pitcher_id: int) -> str:
        if self.roles is None:
            return "unclassified"
        return self.roles.get(pitcher_id, "unclassified")

    def predict_proba(self, history: PitcherWorkloadHistory, pitcher_id: int, as_of_date, team: str | None = None) -> float:
        """`team` is only used (and only needed) when the closer path is
        "closer_only_logistic_regression" -- without it, or without
        team_save_history set, TEAM_SAVE_OPPORTUNITY_FEATURE_NAME falls back
        to 0.0 rather than raising, same "graceful, not silent-wrong"
        spirit as workload_features_for's own sentinels."""
        cutoff_ns = pd.Timestamp(as_of_date).value

        if self.closer_model is not None and self._role_for(pitcher_id) == "closer":
            base = workload_features_for(history, pitcher_id, cutoff_ns)
            if self.closer_kind == "role_aware_logistic_regression":
                save_recency, light_recency = closer_recency_features_for(history, pitcher_id, cutoff_ns)
                row = pd.DataFrame(
                    [{**dict(zip(WORKLOAD_FEATURE_NAMES, base)), "role": "closer",
                      CLOSER_RECENCY_FEATURE_NAMES[0]: save_recency, CLOSER_RECENCY_FEATURE_NAMES[1]: light_recency}]
                )
            else:
                team_feature = 0.0
                if team is not None and self.team_save_history is not None:
                    team_feature = team_save_opportunities_trailing_for(self.team_save_history, team, cutoff_ns)
                row = pd.DataFrame(
                    [{**dict(zip(WORKLOAD_FEATURE_NAMES, base)), TEAM_SAVE_OPPORTUNITY_FEATURE_NAME: team_feature}]
                )
            return float(_closer_predict_proba(self.closer_kind, self.closer_model, row)[0])

        features = workload_features_for(history, pitcher_id, cutoff_ns)
        row = pd.DataFrame([features], columns=WORKLOAD_FEATURE_NAMES)
        return float(_predict_proba(self.model, row)[0])

    def predict_proba_batch(self, examples: pd.DataFrame) -> np.ndarray:
        """If this predictor has a dedicated closer path, `examples` must
        already carry a "role" column (see attach_roles) and, depending on
        which closer path won: the CLOSER_RECENCY_FEATURE_NAMES columns for
        role-aware (see attach_closer_recency_features), or
        TEAM_SAVE_OPPORTUNITY_FEATURE_NAME for closer-only (see
        attach_team_save_opportunity_feature) -- this method never
        recomputes features from raw history for a whole batch, only
        predict_proba's single-query path does that."""
        if self.closer_model is None:
            return _predict_proba(self.model, examples)

        is_closer = (examples["role"] == "closer").to_numpy()
        probs = np.empty(len(examples), dtype="float64")
        if (~is_closer).any():
            probs[~is_closer] = _predict_proba(self.model, examples.loc[~is_closer])
        if is_closer.any():
            probs[is_closer] = _closer_predict_proba(self.closer_kind, self.closer_model, examples.loc[is_closer])
        return probs


@dataclass
class ModelSelectionResult:
    predictor: BullpenAvailabilityPredictor
    general_kind: str
    heuristic_metrics: dict[str, float]
    logistic_metrics: dict[str, float]
    closer_kind: str
    general_model_closer_metrics: dict[str, float]
    closer_only_metrics: dict[str, float]
    role_aware_closer_metrics: dict[str, float]
    general_model_closer_calibration: pd.DataFrame
    closer_only_calibration: pd.DataFrame
    role_aware_closer_calibration: pd.DataFrame


def select_availability_model(
    train_examples: pd.DataFrame,
    val_examples: pd.DataFrame,
    roles: pd.Series,
    config: BullpenAvailabilityConfig,
    n_calibration_bins: int = 10,
    team_save_history: "TeamSaveOpportunityHistory | None" = None,
) -> ModelSelectionResult:
    """`train_examples`/`val_examples` must already carry a "role" column
    (attach_roles), the CLOSER_RECENCY_FEATURE_NAMES columns
    (attach_closer_recency_features), and TEAM_SAVE_OPPORTUNITY_FEATURE_NAME
    (attach_team_save_opportunity_feature) on top of build_query_examples'
    base output. `team_save_history` is only used to attach to the returned
    predictor (see BullpenAvailabilityPredictor.predict_proba) for its
    single-query path -- fitting/evaluation here only ever reads the
    already-attached column, never recomputes it.

    Two decisions, each made honestly on held-out (`val_examples`) data:

    1. General model: heuristic vs. plain logistic regression, whichever
       wins on validation AUC (unchanged from before this module had a
       closer-specific path at all).
    2. Closer path: closer-only logistic regression (CLOSER_ONLY_FEATURE_NAMES
       -- the usual workload features plus TEAM_SAVE_OPPORTUNITY_FEATURE_NAME
       -- fit on just the classified closers' own train examples) vs.
       RoleAwareLogisticRegressionModel (fit on ALL train examples, with
       closer-specific coefficients -- see its docstring), whichever
       calibrates better on held-out closer examples by mean
       |calibration_gap| across n_calibration_bins deciles. The closer-only
       fit is tried first and used whenever it calibrates at least as well;
       the role-aware model exists specifically for the case where ~33
       closers' own data is too thin to calibrate well in isolation.

    The returned predictor's `.model`/`.kind` (general) and
    `.closer_model`/`.closer_kind` (closer path) together cover every role;
    BullpenAvailabilityPredictor.predict_proba(_batch) dispatches between
    them by role automatically.
    """
    heuristic = HeuristicAvailabilityModel()
    general_logistic = fit_logistic_regression_model(train_examples)

    heuristic_val_prob = _predict_proba(heuristic, val_examples)
    logistic_val_prob = _predict_proba(general_logistic, val_examples)
    heuristic_metrics = evaluate_predictions(val_examples["label"].to_numpy(), heuristic_val_prob)
    logistic_metrics = evaluate_predictions(val_examples["label"].to_numpy(), logistic_val_prob)

    if logistic_metrics["auc"] > heuristic_metrics["auc"]:
        general_kind, general_model = "logistic_regression", general_logistic
        logger.info(
            "General model: logistic regression wins on validation AUC (%.4f vs heuristic %.4f).",
            logistic_metrics["auc"], heuristic_metrics["auc"],
        )
    else:
        general_kind, general_model = "heuristic", heuristic
        logger.info(
            "General model: heuristic wins or ties on validation AUC (%.4f vs logistic regression %.4f).",
            heuristic_metrics["auc"], logistic_metrics["auc"],
        )

    train_closer = train_examples[train_examples["role"] == "closer"].reset_index(drop=True)
    val_closer = val_examples[val_examples["role"] == "closer"].reset_index(drop=True)
    logger.info("Closer path: %d train examples, %d val examples.", len(train_closer), len(val_closer))

    if len(train_closer) == 0 or len(val_closer) == 0:
        logger.warning(
            "No classified closers in train and/or val examples -- skipping the closer-specific path "
            "entirely; the general model covers every candidate instead."
        )
        empty_metrics: dict[str, float] = {}
        empty_calibration = pd.DataFrame(columns=["n", "mean_predicted", "observed_rate", "calibration_gap"])
        predictor = BullpenAvailabilityPredictor(
            kind=general_kind, model=general_model, config=config, roles=roles, team_save_history=team_save_history
        )
        return ModelSelectionResult(
            predictor=predictor,
            general_kind=general_kind,
            heuristic_metrics=heuristic_metrics,
            logistic_metrics=logistic_metrics,
            closer_kind="none",
            general_model_closer_metrics=empty_metrics,
            closer_only_metrics=empty_metrics,
            role_aware_closer_metrics=empty_metrics,
            general_model_closer_calibration=empty_calibration,
            closer_only_calibration=empty_calibration,
            role_aware_closer_calibration=empty_calibration,
        )

    general_model_closer_prob = _predict_proba(general_model, val_closer)
    general_model_closer_metrics = evaluate_predictions(val_closer["label"].to_numpy(), general_model_closer_prob)
    general_model_closer_calibration = compute_calibration_table(val_closer["label"].to_numpy(), general_model_closer_prob, n_calibration_bins)

    closer_only_model = fit_logistic_regression_model(train_closer, CLOSER_ONLY_FEATURE_NAMES)
    closer_only_prob = _closer_predict_proba("closer_only_logistic_regression", closer_only_model, val_closer)
    closer_only_metrics = evaluate_predictions(val_closer["label"].to_numpy(), closer_only_prob)
    closer_only_calibration = compute_calibration_table(val_closer["label"].to_numpy(), closer_only_prob, n_calibration_bins)

    role_aware_model = RoleAwareLogisticRegressionModel().fit(train_examples)
    role_aware_prob = role_aware_model.predict_proba(val_closer)
    role_aware_closer_metrics = evaluate_predictions(val_closer["label"].to_numpy(), role_aware_prob)
    role_aware_closer_calibration = compute_calibration_table(val_closer["label"].to_numpy(), role_aware_prob, n_calibration_bins)

    closer_only_gap = mean_abs_calibration_gap(closer_only_calibration)
    role_aware_gap = mean_abs_calibration_gap(role_aware_closer_calibration)
    general_gap = mean_abs_calibration_gap(general_model_closer_calibration)
    logger.info(
        "Closer calibration (mean |gap| across deciles): general=%.4f closer_only=%.4f role_aware=%.4f",
        general_gap, closer_only_gap, role_aware_gap,
    )

    if closer_only_gap <= role_aware_gap:
        closer_kind, closer_model = "closer_only_logistic_regression", closer_only_model
        logger.info(
            "Closer-only logistic regression calibrates at least as well as the role-aware model "
            "(mean |gap| %.4f vs %.4f) -- using it for closers.", closer_only_gap, role_aware_gap,
        )
    else:
        closer_kind, closer_model = "role_aware_logistic_regression", role_aware_model
        logger.info(
            "Closer-only logistic regression's small sample doesn't calibrate as well as the shared "
            "role-aware model (mean |gap| %.4f vs %.4f) -- using the role-aware model for closers instead.",
            closer_only_gap, role_aware_gap,
        )

    predictor = BullpenAvailabilityPredictor(
        kind=general_kind, model=general_model, config=config,
        closer_kind=closer_kind, closer_model=closer_model, roles=roles,
        team_save_history=team_save_history,
    )

    return ModelSelectionResult(
        predictor=predictor,
        general_kind=general_kind,
        heuristic_metrics=heuristic_metrics,
        logistic_metrics=logistic_metrics,
        closer_kind=closer_kind,
        general_model_closer_metrics=general_model_closer_metrics,
        closer_only_metrics=closer_only_metrics,
        role_aware_closer_metrics=role_aware_closer_metrics,
        general_model_closer_calibration=general_model_closer_calibration,
        closer_only_calibration=closer_only_calibration,
        role_aware_closer_calibration=role_aware_closer_calibration,
    )


def save_predictor(predictor: BullpenAvailabilityPredictor, path: Path = DEFAULT_CHECKPOINT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(predictor, f)


def load_predictor(path: Path = DEFAULT_CHECKPOINT_PATH) -> BullpenAvailabilityPredictor:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and validate a bullpen-availability predictor.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--pitches-dir", type=Path, default=PROCESSED_DATA_DIR / "pitches")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--games-dir", type=Path, default=GAMES_DIR)
    parser.add_argument("--pitcher-appearances-dir", type=Path, default=PITCHER_APPEARANCES_DIR)
    parser.add_argument("--batter-appearances-dir", type=Path, default=BATTER_APPEARANCES_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--calibration-plot", type=Path, default=DEFAULT_CALIBRATION_PLOT_PATH)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--min-role-appearances", type=int, default=MIN_APPEARANCES_FOR_ROLE)
    parser.add_argument(
        "--train-season-start", type=int, default=TRAIN_SEASON_RANGE[0],
        help="Overrides the project-wide default train split start (statcast_common.TRAIN_SEASON_RANGE) -- "
        "e.g. for walk-forward retraining at a later season boundary.",
    )
    parser.add_argument(
        "--train-season-end", type=int, default=TRAIN_SEASON_RANGE[1],
        help="Overrides the project-wide default train split end (statcast_common.TRAIN_SEASON_RANGE).",
    )
    parser.add_argument(
        "--val-seasons", type=int, nargs="+", default=list(VAL_SEASONS),
        help="Overrides the project-wide default validation season(s) (statcast_common.VAL_SEASONS).",
    )
    parser.add_argument(
        "--appearance-season-end", type=int, default=None,
        help="Overrides how far forward pitcher-appearance history is built/read (see load_game_split) -- "
        "defaults to the last of --val-seasons. Set past that for walk-forward retraining, so a test-season "
        "game's rest-day lookup can see that pitcher's own recent (also test-season) appearances.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config = BullpenAvailabilityConfig.from_yaml(args.config)
    train_season_range = (args.train_season_start, args.train_season_end)
    val_seasons = tuple(args.val_seasons)

    logger.info("Loading pitches and appearance history...")
    full_pitches = read_partitioned(args.pitches_dir)
    pitches = full_pitches[full_pitches["is_valid"]].reset_index(drop=True)
    pitch_counts = compute_pitch_counts(pitches)
    entry_situations = compute_entry_situations(pitches)

    _, _, pitcher_appearances, _ = load_game_split(
        raw_dir=args.raw_dir,
        games_dir=args.games_dir,
        pitcher_appearances_dir=args.pitcher_appearances_dir,
        batter_appearances_dir=args.batter_appearances_dir,
        train_season_range=train_season_range,
        val_seasons=val_seasons,
        appearance_season_end=args.appearance_season_end,
    )

    history = build_workload_history(pitcher_appearances, pitch_counts, entry_situations)
    roles = classify_reliever_roles(
        pitcher_appearances, entry_situations[["game_pk", "pitcher_id", "entry_inning"]], pitch_counts, args.min_role_appearances
    )
    role_counts = roles.value_counts().to_dict()
    logger.info("Classified %d pitchers by role (>= %d relief appearances each): %s", len(roles), args.min_role_appearances, role_counts)

    team_save_counts = compute_team_save_opportunity_counts(pitcher_appearances, entry_situations)
    team_save_history = build_team_save_opportunity_history(team_save_counts)

    logger.info("Building candidate query examples (window=%d days)...", config.window_days)
    examples = build_query_examples(pitcher_appearances, history, config.window_days)
    examples = attach_roles(examples, roles)
    examples = attach_closer_recency_features(examples, history)
    examples = attach_team_save_opportunity_feature(examples, team_save_history)
    logger.info(
        "%d candidate examples (%d appeared, %d did not)",
        len(examples), int(examples["label"].sum()), int((~examples["label"].astype(bool)).sum()),
    )

    train_examples = examples[examples["season"].between(*train_season_range)].reset_index(drop=True)
    val_examples = examples[examples["season"].isin(val_seasons)].reset_index(drop=True)
    logger.info("Train examples: %d, Val examples: %d", len(train_examples), len(val_examples))

    selection = select_availability_model(train_examples, val_examples, roles, config, args.calibration_bins, team_save_history)

    logger.info("General model: %s", selection.general_kind)
    logger.info("  Heuristic validation metrics:          %s", selection.heuristic_metrics)
    logger.info("  Logistic regression validation metrics: %s", selection.logistic_metrics)
    logger.info("Closer path: %s", selection.closer_kind)
    logger.info("  General model on closers only: %s", selection.general_model_closer_metrics)
    logger.info("  Closer-only LR on closers:      %s", selection.closer_only_metrics)
    logger.info("  Role-aware LR on closers:       %s", selection.role_aware_closer_metrics)
    logger.info("Closer calibration (general model):\n%s", selection.general_model_closer_calibration.to_string(index=False))
    logger.info("Closer calibration (closer-only LR):\n%s", selection.closer_only_calibration.to_string(index=False))
    logger.info("Closer calibration (role-aware LR):\n%s", selection.role_aware_closer_calibration.to_string(index=False))

    predictor = selection.predictor
    save_predictor(predictor, args.checkpoint)
    logger.info("Saved predictor to %s", args.checkpoint)

    logger.info("Rerunning the full calibration check (all roles) against the final composite predictor...")
    val_prob = predictor.predict_proba_batch(val_examples)
    calibration_tables = calibration_by_role(val_examples, val_prob, roles, args.calibration_bins)
    for name, table in calibration_tables.items():
        logger.info("Calibration (%s), mean |gap|=%.4f:\n%s", name, mean_abs_calibration_gap(table), table.to_string(index=False))

    plot_calibration_reliability(calibration_tables, args.calibration_plot)
    logger.info("Saved calibration reliability plot to %s", args.calibration_plot)


if __name__ == "__main__":
    main()
