"""Hazard-style model: predicts, batter by batter within a pitcher's
current stint, the probability that pitcher is removed after the CURRENT
batter -- a discrete-time survival/hazard model over "batters faced this
stint," trained directly on real, observed removal points. Unlike
bullpen_availability.py's "available" (never directly observed, only
proxied by "did they appear"), a pitching change is directly visible in the
pitch data: the point where `pitcher_id` changes between one at-bat and the
next, for the same team's pitching sequence, within the same game. No proxy
needed here.

Isolating "the same team's pitching sequence" matters: naively watching for
`pitcher_id` to differ from the previous pitch-table row would also fire at
every half-inning switch, since the two teams' pitchers alternate between
Top and Bottom half-innings -- that's not a substitution, just the other
team's turn on defense. build_hook_examples splits each game into its two
teams' own chronological at-bat sequences (via inning_topbot) and only
counts a pitcher_id change *within* one team's own sequence as a real hook.

Censoring: the very last pitcher a team uses in a game was never actually
"not removed" at a decision point -- the game just ended before a decision
was needed. That final stint's last batter is dropped entirely (neither
label=1 nor label=0), the standard right-censoring treatment; every batter
before it, including every other pitcher's full stint that game, is a
genuine, directly observed data point.

Features (HOOK_FEATURE_NAMES): batters faced so far this stint (including
the current one), cumulative pitch count this stint, run differential from
the pitcher's own team's perspective, whether a runner is on base, and that
pitcher's own personalized "hook prior" -- their expanding-window average
batters-faced/pitch-count at removal from their own strictly-prior stints
(league-wide average, from the train split only, for a pitcher's first
recorded stint). Run differential and the on-base flag are read from the
*next* at-bat's first pitch (the true post-plate-appearance game state --
any runs/baserunners this at-bat produced are now reflected), not the
current at-bat's own last pitch, which would still be missing whatever
this at-bat itself just did.

A backtest split by role found the pooled model above is dramatically worse
for starters than relievers (~5.7x the batters-faced error) while being
essentially unaffected by how close the game is -- pointing at something
structural about pooling two very differently-shaped removal processes into
one hazard curve, not a missing bullpen-availability-style feature. This
module fits *separate* hazard models per role instead of one pooled model
(fit_and_compare_role_specific_models): a starter model on
STARTER_FEATURE_NAMES, which adds pitch-count milestone indicators
(pitch_count_ge_75/90/100/110/120 -- real managers' hook decisions are
widely understood to hinge on round pitch-count numbers, which a single
linear pitch_count coefficient can't represent) and times_through_order
(already computed in the processed pitch table -- the times-through-the-
order penalty is a well-established driver of starter removal specifically,
essentially meaningless for a reliever who rarely faces a batter twice) --
cutting starters' batters-faced MAE by more than half (5.04 -> 2.12).

A plain reliever-only split (same HOOK_FEATURE_NAMES, just fit on relief
stints alone) made relievers slightly *worse* (0.88 -> 1.03 batters-faced
MAE): splitting away from starters cost it training data without adding any
new signal, since relievers never had the missing-feature problem starters
did. The model actually shipped in the "reliever" slot is instead a
*hybrid*: trained on the full combined population (starters and relievers
together, same as pooled) but with the same expanded STARTER_FEATURE_NAMES
feature set -- mostly inert for a real relief outing (a reliever rarely
reaches pitch_count_ge_75, let alone higher milestones, and rarely faces a
batter a second time) but free to use if informative, while keeping
pooled's full training population. Run this module (or see its own logged
comparison output) for the actual, current pooled/reliever_only/hybrid
numbers -- deliberately not hardcoded here, since they'd silently go stale
the next time this data or these models change.

Each role's personalized "hook prior" is scoped to match what its model was
actually trained on: the starter model uses that pitcher's own history
*as a starter* only (a swingman's starts and relief outings aren't the same
distribution); the hybrid reliever model, trained on the combined
population, uses the same role-blind history the original pooled model did.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.data.statcast_common import PROCESSED_DATA_DIR, TRAIN_SEASON_RANGE, VAL_SEASONS, read_partitioned

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = Path("checkpoints") / "hook_model.pkl"
NS_PER_DAY = 86_400_000_000_000

HOOK_FEATURE_NAMES = [
    "batters_faced_so_far",
    "pitch_count",
    "run_differential",
    "runner_on_base",
    "historical_avg_batters_faced_at_removal",
    "historical_avg_pitch_count_at_removal",
]

# Well-known informal pitch-count hook thresholds. Reliever stints
# essentially never reach these, which is exactly why the indicators built
# from them (see attach_pitch_count_milestones) are starter-only rather
# than added to HOOK_FEATURE_NAMES for everyone.
PITCH_COUNT_MILESTONES = [75, 90, 100, 110, 120]
PITCH_COUNT_MILESTONE_FEATURE_NAMES = [f"pitch_count_ge_{m}" for m in PITCH_COUNT_MILESTONES]

# The starter-only hazard model's feature set: everything in
# HOOK_FEATURE_NAMES, plus the pitch-count milestones and times_through_order
# (already computed in the processed pitch table -- see build_hook_examples)
# hypothesized to be the real drivers of starter removal specifically.
STARTER_FEATURE_NAMES = HOOK_FEATURE_NAMES + PITCH_COUNT_MILESTONE_FEATURE_NAMES + ["times_through_order"]


# ---------------------------------------------------------------------------
# Stint reconstruction: one row per (pitcher's stint, batter faced), with
# the real, observed removal label.
# ---------------------------------------------------------------------------


def _at_bat_table(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, at_bat_number): the pitcher/batter who were
    involved, how many pitches were thrown, and the game state (score,
    baserunners) as of that at-bat's first pitch -- used both directly (a
    later at-bat's "first pitch" state is the *previous* at-bat's
    post-plate-appearance state) and aggregated (pitch count)."""
    sorted_pitches = pitches.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    first = sorted_pitches.drop_duplicates(subset=["game_pk", "at_bat_number"], keep="first")
    counts = sorted_pitches.groupby(["game_pk", "at_bat_number"], as_index=False).size().rename(columns={"size": "n_pitches"})

    at_bats = first[
        ["game_pk", "at_bat_number", "game_date", "season", "inning_topbot", "pitcher_id", "batter_id",
         "home_team", "away_team", "home_score", "away_score", "on_1b", "on_2b", "on_3b", "times_through_order"]
    ].merge(counts, on=["game_pk", "at_bat_number"])
    at_bats = at_bats.sort_values(["game_pk", "at_bat_number"]).reset_index(drop=True)

    # game_pk/at_bat_number/pitcher_id are pandas nullable Int64 on the real
    # processed table (never actually null on is_valid rows, but the dtype
    # itself is nullable) -- comparisons against a .shift()-introduced NA at
    # a group's first/last row then produce nullable *booleans* (pd.NA, not
    # True/False), which silently poison every downstream & / | / ~ and
    # finally crashes .astype(int). Casting to plain numpy int64 up front
    # (safe: these columns are guaranteed non-null here) keeps every
    # shift-based comparison in this module in plain bool/numpy land.
    for col in ("game_pk", "at_bat_number", "pitcher_id", "batter_id"):
        at_bats[col] = at_bats[col].astype("int64")
    # times_through_order feeds a model (as a plain numeric feature, not a
    # shift-based comparison) but sklearn needs a real numeric dtype, not
    # pandas' nullable Int64 extension type.
    at_bats["times_through_order"] = at_bats["times_through_order"].astype("float64")
    return at_bats


def build_hook_examples(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per (pitcher stint, batter faced) with the base features
    (everything in HOOK_FEATURE_NAMES except the two personalized
    historical-average columns, which attach_removal_history_features adds
    afterward) and the real, observed removal label. Rows belonging to a
    team's *final* pitching stint of a game are dropped (right-censored --
    see module docstring), not labeled 0.
    """
    at_bats = _at_bat_table(pitches)

    is_away_batting = at_bats["inning_topbot"] == "Top"
    at_bats["pitcher_team"] = np.where(is_away_batting, at_bats["home_team"], at_bats["away_team"])

    at_bats = at_bats.sort_values(["game_pk", "pitcher_team", "at_bat_number"]).reset_index(drop=True)

    same_team_game = (at_bats["game_pk"] == at_bats["game_pk"].shift()) & (at_bats["pitcher_team"] == at_bats["pitcher_team"].shift())
    new_stint = ~same_team_game | (at_bats["pitcher_id"] != at_bats["pitcher_id"].shift())
    at_bats["stint_id"] = new_stint.cumsum()

    # A team's first stint of the game (lowest stint_id in its own
    # (game_pk, pitcher_team) group) is the start; every later stint for
    # that team is relief -- directly derived from stint order, no separate
    # box-score "who started" lookup needed.
    first_stint_id = at_bats.groupby(["game_pk", "pitcher_team"])["stint_id"].transform("min")
    at_bats["is_starter"] = at_bats["stint_id"] == first_stint_id

    at_bats["batters_faced_so_far"] = at_bats.groupby("stint_id").cumcount() + 1
    at_bats["pitch_count"] = at_bats.groupby("stint_id")["n_pitches"].cumsum()

    next_same_stint = at_bats["stint_id"] == at_bats["stint_id"].shift(-1)
    is_last_of_stint = ~next_same_stint
    next_same_team_game = (at_bats["game_pk"] == at_bats["game_pk"].shift(-1)) & (
        at_bats["pitcher_team"] == at_bats["pitcher_team"].shift(-1)
    )
    is_censored = is_last_of_stint & ~next_same_team_game
    at_bats["label"] = (is_last_of_stint & ~is_censored).astype(int)

    # Post-plate-appearance game state: the *next* at-bat's own first-pitch
    # state, in the whole game's chronological order (not scoped to this
    # team's own sequence -- the other team's next at-bat still reflects
    # this at-bat's runs/baserunners just as validly).
    global_order = at_bats.sort_values(["game_pk", "at_bat_number"]).reset_index(drop=True)
    next_valid = global_order["game_pk"] == global_order["game_pk"].shift(-1)
    next_state = pd.DataFrame(
        {
            "game_pk": global_order["game_pk"],
            "at_bat_number": global_order["at_bat_number"],
            "next_valid": next_valid,
            "next_home_score": global_order["home_score"].shift(-1),
            "next_away_score": global_order["away_score"].shift(-1),
            "next_on_1b": global_order["on_1b"].shift(-1),
            "next_on_2b": global_order["on_2b"].shift(-1),
            "next_on_3b": global_order["on_3b"].shift(-1),
        }
    )
    at_bats = at_bats.merge(next_state, on=["game_pk", "at_bat_number"], how="left")

    is_pitcher_team_home = at_bats["pitcher_team"] == at_bats["home_team"]
    pitcher_score = at_bats["next_home_score"].where(is_pitcher_team_home, at_bats["next_away_score"])
    opponent_score = at_bats["next_away_score"].where(is_pitcher_team_home, at_bats["next_home_score"])
    at_bats["run_differential"] = (pitcher_score - opponent_score).astype("float64")
    at_bats["runner_on_base"] = (
        at_bats["next_on_1b"].notna() | at_bats["next_on_2b"].notna() | at_bats["next_on_3b"].notna()
    ).astype(float)

    # Drop censored stints' final batter (no observed decision) and the
    # handful of rows with no next at-bat at all (only the game's literal
    # final at-bat, already covered by is_censored, but guarded directly too).
    keep = (~is_censored) & at_bats["next_valid"]
    result = at_bats.loc[
        keep,
        ["game_pk", "game_date", "season", "pitcher_id", "pitcher_team", "stint_id", "is_starter",
         "batters_faced_so_far", "pitch_count", "run_differential", "runner_on_base", "times_through_order", "label"],
    ].reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Personalized "hook prior": a pitcher's own expanding-window average
# batters-faced/pitch-count at removal, strictly before a given game date.
# ---------------------------------------------------------------------------


@dataclass
class PitcherRemovalHistory:
    dates_by_pitcher: dict[int, np.ndarray]  # sorted int64 date_ns arrays, one per completed stint
    batters_faced_by_pitcher: dict[int, np.ndarray]  # parallel: batters faced when that stint ended
    pitch_count_by_pitcher: dict[int, np.ndarray]  # parallel: pitches thrown when that stint ended
    league_avg_batters_faced: float
    league_avg_pitch_count: float


def build_removal_history(examples: pd.DataFrame, league_avg_examples: pd.DataFrame) -> PitcherRemovalHistory:
    """`examples`: build_hook_examples' output (any split -- the per-pitcher
    expanding-mean lookup is leak-safe by construction, same "strictly
    before cutoff" reasoning as the rest of this project's history
    lookups). `league_avg_examples` should be train-split only: it seeds
    the fallback for a pitcher with no prior recorded removal at all, and
    using val-period data there would leak a small amount of future
    information into exactly the cold-start predictions that fallback
    exists for.
    """
    completed = examples[examples["label"] == 1].sort_values(["pitcher_id", "game_date"])

    dates_by_pitcher: dict[int, np.ndarray] = {}
    batters_faced_by_pitcher: dict[int, np.ndarray] = {}
    pitch_count_by_pitcher: dict[int, np.ndarray] = {}
    for pitcher_id, group in completed.groupby("pitcher_id"):
        dates_by_pitcher[pitcher_id] = group["game_date"].to_numpy().astype("datetime64[ns]").astype("int64")
        batters_faced_by_pitcher[pitcher_id] = group["batters_faced_so_far"].to_numpy(dtype="float64")
        pitch_count_by_pitcher[pitcher_id] = group["pitch_count"].to_numpy(dtype="float64")

    league_completed = league_avg_examples[league_avg_examples["label"] == 1]
    return PitcherRemovalHistory(
        dates_by_pitcher, batters_faced_by_pitcher, pitch_count_by_pitcher,
        league_avg_batters_faced=float(league_completed["batters_faced_so_far"].mean()),
        league_avg_pitch_count=float(league_completed["pitch_count"].mean()),
    )


def removal_history_features_for(history: PitcherRemovalHistory, pitcher_id: int, cutoff_ns: int) -> tuple[float, float]:
    """(historical_avg_batters_faced_at_removal, historical_avg_pitch_count_at_removal)
    for `pitcher_id`, averaged over their own completed stints strictly
    before `cutoff_ns`. League-wide average (see build_removal_history) for
    a pitcher with no such prior stint yet."""
    dates = history.dates_by_pitcher.get(pitcher_id)
    if dates is None or len(dates) == 0:
        return history.league_avg_batters_faced, history.league_avg_pitch_count
    end = int(np.searchsorted(dates, cutoff_ns, side="left"))
    if end == 0:
        return history.league_avg_batters_faced, history.league_avg_pitch_count
    return (
        float(history.batters_faced_by_pitcher[pitcher_id][:end].mean()),
        float(history.pitch_count_by_pitcher[pitcher_id][:end].mean()),
    )


def attach_removal_history_features(examples: pd.DataFrame, history: PitcherRemovalHistory) -> pd.DataFrame:
    """Adds historical_avg_batters_faced_at_removal / historical_avg_pitch_count_at_removal
    from a single, role-blind `history`. Every row of the same (pitcher_id,
    game_date) stint gets the same values (the personalized prior is fixed
    as of that game's date, not recomputed batter to batter within it).
    See attach_removal_history_features_by_role for the role-aware version
    the starter/reliever-specific models actually use."""
    examples = examples.copy()
    stint_keys = examples[["pitcher_id", "game_date"]].drop_duplicates()
    avg_batters = np.empty(len(stint_keys))
    avg_pitches = np.empty(len(stint_keys))
    for i, (pitcher_id, game_date) in enumerate(zip(stint_keys["pitcher_id"], stint_keys["game_date"])):
        avg_batters[i], avg_pitches[i] = removal_history_features_for(history, pitcher_id, pd.Timestamp(game_date).value)
    stint_keys = stint_keys.copy()
    stint_keys["historical_avg_batters_faced_at_removal"] = avg_batters
    stint_keys["historical_avg_pitch_count_at_removal"] = avg_pitches
    return examples.merge(stint_keys, on=["pitcher_id", "game_date"], how="left")


def attach_removal_history_features_by_role(
    examples: pd.DataFrame, starter_history: PitcherRemovalHistory, reliever_history: PitcherRemovalHistory
) -> pd.DataFrame:
    """Same idea as attach_removal_history_features, but looks each stint's
    personalized prior up in starter_history or reliever_history depending
    on that stint's own is_starter flag -- a swingman's average removal
    point as a starter and as a reliever aren't the same distribution, so
    blending them into one number would make a worse prior for either role.
    """
    examples = examples.copy()
    stint_keys = examples[["pitcher_id", "game_date", "is_starter"]].drop_duplicates()
    avg_batters = np.empty(len(stint_keys))
    avg_pitches = np.empty(len(stint_keys))
    for i, (pitcher_id, game_date, is_starter) in enumerate(
        zip(stint_keys["pitcher_id"], stint_keys["game_date"], stint_keys["is_starter"])
    ):
        history = starter_history if is_starter else reliever_history
        avg_batters[i], avg_pitches[i] = removal_history_features_for(history, pitcher_id, pd.Timestamp(game_date).value)
    stint_keys = stint_keys.copy()
    stint_keys["historical_avg_batters_faced_at_removal"] = avg_batters
    stint_keys["historical_avg_pitch_count_at_removal"] = avg_pitches
    return examples.merge(stint_keys, on=["pitcher_id", "game_date", "is_starter"], how="left")


def attach_pitch_count_milestones(examples: pd.DataFrame) -> pd.DataFrame:
    """Binary indicators for crossing well-known informal pitch-count hook
    thresholds (PITCH_COUNT_MILESTONES) -- starter removal is widely
    understood to hinge on round-number pitch counts (100 especially) in a
    way a single linear pitch_count coefficient can't represent."""
    examples = examples.copy()
    for name, threshold in zip(PITCH_COUNT_MILESTONE_FEATURE_NAMES, PITCH_COUNT_MILESTONES):
        examples[name] = (examples["pitch_count"] >= threshold).astype(float)
    return examples


# ---------------------------------------------------------------------------
# The hazard model itself.
# ---------------------------------------------------------------------------


def fit_hook_model(train_examples: pd.DataFrame, feature_names: list[str] = HOOK_FEATURE_NAMES) -> LogisticRegression:
    """feature_names defaults to HOOK_FEATURE_NAMES (pooled/reliever
    models); pass STARTER_FEATURE_NAMES for the starter model, which needs
    `train_examples` to already carry the pitch-count milestone columns
    (attach_pitch_count_milestones) and times_through_order (already on
    build_hook_examples' own output)."""
    model = LogisticRegression(max_iter=1000)
    model.fit(train_examples[feature_names].to_numpy(), train_examples["label"].to_numpy())
    return model


def predict_hazard(model: LogisticRegression, examples: pd.DataFrame, feature_names: list[str] = HOOK_FEATURE_NAMES) -> np.ndarray:
    return model.predict_proba(examples[feature_names].to_numpy())[:, 1]


def evaluate_hazard_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    predicted_removed = y_prob >= 0.5
    positive = y_true == 1
    return {
        "n": len(y_true),
        "n_removed": int(positive.sum()),
        "auc": float(roc_auc_score(y_true, y_prob)) if positive.any() and (~positive).any() else float("nan"),
        "accuracy": float((predicted_removed == positive).mean()),
        "brier_score": float(np.mean((y_prob - y_true) ** 2)),
    }


# ---------------------------------------------------------------------------
# Backtest: how close does the model's implied removal point land to the
# real one?
# ---------------------------------------------------------------------------


def expected_removal_point(
    model: LogisticRegression, stint_examples: pd.DataFrame, feature_names: list[str] = HOOK_FEATURE_NAMES
) -> tuple[float, float]:
    """`stint_examples`: one pitcher-stint's rows, sorted by
    batters_faced_so_far (every batter actually faced that stint, in
    order). Returns (expected_batters_faced_at_removal,
    expected_pitch_count_at_removal): the survival-weighted expectation of
    the hazard model's own removal-probability curve, evaluated only over
    the batters actually observed in this stint (there's no real feature
    data past what actually happened to evaluate further). Any probability
    mass the curve never assigns within that observed window ("survived"
    every batter actually faced) is attached to the stint's own last
    batter/pitch count -- the standard survival-analysis convention of
    censoring an expectation at the observation horizon, and the only
    sensible finite answer when the model's implied hook never actually
    crosses within the window we can evaluate it over.
    """
    hazard = predict_hazard(model, stint_examples, feature_names)
    prior_survival = np.concatenate([[1.0], np.cumprod(1 - hazard)[:-1]])
    p_removed_here = prior_survival * hazard

    batters = stint_examples["batters_faced_so_far"].to_numpy(dtype="float64")
    pitches = stint_examples["pitch_count"].to_numpy(dtype="float64")
    residual = max(0.0, 1.0 - p_removed_here.sum())

    expected_batters = float((p_removed_here * batters).sum() + residual * batters[-1])
    expected_pitches = float((p_removed_here * pitches).sum() + residual * pitches[-1])
    return expected_batters, expected_pitches


def backtest_removal_point(
    model: LogisticRegression, test_examples: pd.DataFrame, feature_names: list[str] = HOOK_FEATURE_NAMES
) -> pd.DataFrame:
    """One row per held-out stint: the real, observed removal point
    (batters faced / pitch count -- by construction, exactly the stint's
    last row, since every stint here ends in a genuine, non-censored
    removal) versus the model's expected_removal_point, and their errors.
    Directly answers "how close does the predicted removal point land to
    the actual one" -- see also summarize_removal_point_errors for the
    aggregate MAE this produces per-stint detail for.
    """
    rows = []
    for stint_id, group in test_examples.groupby("stint_id"):
        group = group.sort_values("batters_faced_so_far")
        actual_batters = float(group["batters_faced_so_far"].iloc[-1])
        actual_pitches = float(group["pitch_count"].iloc[-1])
        expected_batters, expected_pitches = expected_removal_point(model, group, feature_names)
        rows.append(
            {
                "stint_id": stint_id,
                "pitcher_id": group["pitcher_id"].iloc[0],
                "game_pk": group["game_pk"].iloc[0],
                "is_starter": bool(group["is_starter"].iloc[0]),
                # Run differential as of the actual removal point -- the
                # game state that was live when the real hook decision got
                # made, for slicing the backtest by "how close was the
                # game at the moment of the decision" (see
                # classify_game_closeness).
                "run_differential_at_removal": float(group["run_differential"].iloc[-1]),
                "actual_batters_faced": actual_batters,
                "predicted_batters_faced": expected_batters,
                "batters_faced_error": expected_batters - actual_batters,
                "actual_pitch_count": actual_pitches,
                "predicted_pitch_count": expected_pitches,
                "pitch_count_error": expected_pitches - actual_pitches,
            }
        )
    return pd.DataFrame(rows)


def summarize_removal_point_errors(backtest_results: pd.DataFrame) -> dict[str, float]:
    return {
        "n_stints": len(backtest_results),
        "batters_faced_mae": float(backtest_results["batters_faced_error"].abs().mean()),
        "batters_faced_bias": float(backtest_results["batters_faced_error"].mean()),
        "pitch_count_mae": float(backtest_results["pitch_count_error"].abs().mean()),
        "pitch_count_bias": float(backtest_results["pitch_count_error"].mean()),
    }


# Whether the game was "close" (a manager weighing the bullpen carefully)
# or a "blowout" (the hook decision matters much less) as of the actual
# removal point -- a run differential of +/-2 or fewer is the standard
# rough cutoff for late-game strategic urgency (roughly a two-run save
# situation's own margin), reused here rather than inventing a new threshold.
CLOSE_GAME_MAX_ABS_RUN_DIFFERENTIAL = 2


def classify_game_closeness(run_differential: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(run_differential.abs() <= CLOSE_GAME_MAX_ABS_RUN_DIFFERENTIAL, "close", "blowout"),
        index=run_differential.index,
    )


def summarize_removal_point_errors_by_group(backtest_results: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """backtest_results: backtest_removal_point's output, with a `group_col`
    column already attached (e.g. "is_starter", or a classify_game_closeness
    column assigned under some name). One row per distinct group value,
    with the same fields summarize_removal_point_errors returns for the
    whole set -- so a uniform bias across groups versus one concentrated in
    a specific situation shows up directly, side by side."""
    rows = []
    for value, subset in backtest_results.groupby(group_col):
        summary = summarize_removal_point_errors(subset)
        summary[group_col] = value
        rows.append(summary)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Role-specific models: does splitting training data by is_starter (rather
# than pooling everyone into one hazard curve) fix the ~5.7x worse
# batters-faced error starters showed under the pooled model?
# ---------------------------------------------------------------------------


def fit_and_compare_role_specific_models(
    examples: pd.DataFrame, train_examples: pd.DataFrame, val_examples: pd.DataFrame
) -> tuple[LogisticRegression, LogisticRegression, PitcherRemovalHistory, PitcherRemovalHistory, pd.DataFrame]:
    """`examples`/`train_examples`/`val_examples`: build_hook_examples'
    output (or a season-filtered slice of it) -- none of these need
    historical-average or milestone columns attached yet; this function
    does that itself, separately for each model, since different models
    below need differently-scoped personalized histories and feature sets.

    Fits four hazard models on `train_examples`:
    - pooled: HOOK_FEATURE_NAMES, every stint, role-blind personalized
      history -- "the original pooled model," the first baseline.
    - starter: STARTER_FEATURE_NAMES (adds pitch-count milestone indicators
      and times_through_order), only is_starter=True stints, personalized
      history from that pitcher's own prior starts only. This is the model
      that actually needed the extra features -- kept exactly as before.
    - reliever_only: HOOK_FEATURE_NAMES, only is_starter=False stints,
      personalized history from that pitcher's own prior relief appearances
      only -- "the reliever-only model," the second baseline. It turned out
      slightly *worse* than pooled for relievers: splitting away from
      starters cost it training data without adding any new signal, since
      relievers didn't have the missing-feature problem starters did.
    - hybrid_reliever: STARTER_FEATURE_NAMES, but trained on *every* stint
      (starters and relievers together, same population as pooled) rather
      than only relief stints, with the same role-blind personalized
      history pooled uses. The fix this function exists to test: does
      giving the reliever-serving model the expanded feature set (mostly
      inert for real relief outings -- a reliever rarely reaches pitch_count_ge_75,
      let alone the higher milestones, and rarely faces a batter a second
      time -- but free to use if informative) while keeping the full
      combined training population recover reliever_only's lost accuracy,
      without giving up dispatch-by-role for the model that actually needed
      it (starters)?

    Backtests all four, each evaluated only on the role it actually applies
    to: pooled-on-starters, starter-on-starters, pooled-on-relievers,
    reliever_only-on-relievers, hybrid_reliever-on-relievers.

    Returns (starter_model, hybrid_reliever_model, starter_history,
    pooled_history, comparison) -- exactly what HookModelPredictor's
    dispatch-by-role architecture ships: the starter-specific model/history
    unchanged, and the *hybrid* reliever model together with the role-blind
    (pooled) history it was actually trained against in the "reliever" slot
    (see HookModelPredictor's docstring for why that slot no longer holds
    the reliever-only model). pooled_model and reliever_only_model are
    fit only to produce their own comparison rows and aren't returned.
    """
    pooled_history = build_removal_history(examples, league_avg_examples=train_examples)
    starter_history = build_removal_history(
        examples[examples["is_starter"]], league_avg_examples=train_examples[train_examples["is_starter"]]
    )
    reliever_history = build_removal_history(
        examples[~examples["is_starter"]], league_avg_examples=train_examples[~train_examples["is_starter"]]
    )

    pooled_train = attach_removal_history_features(train_examples, pooled_history)
    pooled_val = attach_removal_history_features(val_examples, pooled_history)
    pooled_model = fit_hook_model(pooled_train, HOOK_FEATURE_NAMES)

    role_train = attach_removal_history_features_by_role(train_examples, starter_history, reliever_history)
    role_val = attach_removal_history_features_by_role(val_examples, starter_history, reliever_history)
    role_train = attach_pitch_count_milestones(role_train)
    role_val = attach_pitch_count_milestones(role_val)

    starter_train = role_train[role_train["is_starter"]].reset_index(drop=True)
    reliever_only_train = role_train[~role_train["is_starter"]].reset_index(drop=True)
    starter_model = fit_hook_model(starter_train, STARTER_FEATURE_NAMES)
    reliever_only_model = fit_hook_model(reliever_only_train, HOOK_FEATURE_NAMES)

    # Hybrid reliever model: same combined population and role-blind
    # history as pooled, but with pooled_train/pooled_val's rows also
    # carrying the starter-style expanded feature set.
    hybrid_train = attach_pitch_count_milestones(pooled_train)
    hybrid_val = attach_pitch_count_milestones(pooled_val)
    hybrid_reliever_model = fit_hook_model(hybrid_train, STARTER_FEATURE_NAMES)

    comparisons = [
        ("pooled_on_starters", pooled_model, HOOK_FEATURE_NAMES, pooled_val[pooled_val["is_starter"]]),
        ("starter_model_on_starters", starter_model, STARTER_FEATURE_NAMES, role_val[role_val["is_starter"]]),
        ("pooled_on_relievers", pooled_model, HOOK_FEATURE_NAMES, pooled_val[~pooled_val["is_starter"]]),
        ("reliever_only_model_on_relievers", reliever_only_model, HOOK_FEATURE_NAMES, role_val[~role_val["is_starter"]]),
        ("hybrid_reliever_model_on_relievers", hybrid_reliever_model, STARTER_FEATURE_NAMES, hybrid_val[~hybrid_val["is_starter"]]),
    ]
    rows = []
    for label, model, feature_names, val_subset in comparisons:
        backtest_results = backtest_removal_point(model, val_subset.reset_index(drop=True), feature_names)
        summary = summarize_removal_point_errors(backtest_results)
        summary["model"] = label
        rows.append(summary)
    comparison = pd.DataFrame(rows)[["model", "n_stints", "batters_faced_mae", "batters_faced_bias", "pitch_count_mae", "pitch_count_bias"]]

    return starter_model, hybrid_reliever_model, starter_history, pooled_history, comparison


# ---------------------------------------------------------------------------
# Packaged predictor + persistence.
# ---------------------------------------------------------------------------


@dataclass
class HookModelPredictor:
    """Two hazard models -- one per role, see module docstring for why a
    single pooled model was replaced -- plus each role's own personalized
    removal-history lookup.

    Both models expect STARTER_FEATURE_NAMES (the expanded feature set,
    including pitch-count milestones and times_through_order), not just
    starter_model: the "reliever" slot holds a *hybrid* model trained on
    the full combined (starter + reliever) population with that same
    expanded feature set, not a reliever-only model on the narrower
    HOOK_FEATURE_NAMES -- a plain reliever-only refit turned out slightly
    worse than the original pooled model for relievers (it lost training
    data without gaining any new signal, since relievers didn't have the
    missing-feature problem starters did), while this hybrid version keeps
    the full training population *and* offers the extra features to every
    role, mostly inert for real relief outings but free to use if
    informative. reliever_history is correspondingly the same role-blind
    (pooled) history the hybrid model was actually trained against, not a
    relief-only average -- see fit_and_compare_role_specific_models.
    """

    starter_model: LogisticRegression
    reliever_model: LogisticRegression
    starter_history: PitcherRemovalHistory
    reliever_history: PitcherRemovalHistory

    def predict_proba(
        self,
        pitcher_id: int,
        as_of_date,
        is_starter: bool,
        batters_faced_so_far: int,
        pitch_count: int,
        run_differential: float,
        runner_on_base: bool,
        times_through_order: int = 0,
    ) -> float:
        cutoff_ns = pd.Timestamp(as_of_date).value
        history = self.starter_history if is_starter else self.reliever_history
        avg_batters, avg_pitches = removal_history_features_for(history, pitcher_id, cutoff_ns)
        milestones = {
            name: float(pitch_count >= threshold)
            for name, threshold in zip(PITCH_COUNT_MILESTONE_FEATURE_NAMES, PITCH_COUNT_MILESTONES)
        }
        row = pd.DataFrame(
            [
                {
                    "batters_faced_so_far": batters_faced_so_far,
                    "pitch_count": pitch_count,
                    "run_differential": run_differential,
                    "runner_on_base": float(runner_on_base),
                    "historical_avg_batters_faced_at_removal": avg_batters,
                    "historical_avg_pitch_count_at_removal": avg_pitches,
                    **milestones,
                    "times_through_order": float(times_through_order),
                }
            ]
        )
        model = self.starter_model if is_starter else self.reliever_model
        return float(predict_hazard(model, row, STARTER_FEATURE_NAMES)[0])

    def predict_proba_batch(self, examples: pd.DataFrame) -> np.ndarray:
        """`examples` must have an "is_starter" column and all of
        STARTER_FEATURE_NAMES -- both models expect the same expanded
        feature set now (see class docstring)."""
        is_starter = examples["is_starter"].to_numpy()
        probs = np.empty(len(examples), dtype="float64")
        if is_starter.any():
            probs[is_starter] = predict_hazard(self.starter_model, examples.loc[is_starter], STARTER_FEATURE_NAMES)
        if (~is_starter).any():
            probs[~is_starter] = predict_hazard(self.reliever_model, examples.loc[~is_starter], STARTER_FEATURE_NAMES)
        return probs


def save_predictor(predictor: HookModelPredictor, path: Path = DEFAULT_CHECKPOINT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(predictor, f)


def load_predictor(path: Path = DEFAULT_CHECKPOINT_PATH) -> HookModelPredictor:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and backtest the pitcher-hook hazard model.")
    parser.add_argument("--pitches-dir", type=Path, default=PROCESSED_DATA_DIR / "pitches")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    logger.info("Loading pitches...")
    full_pitches = read_partitioned(args.pitches_dir)
    pitches = full_pitches[full_pitches["is_valid"]].reset_index(drop=True)

    logger.info("Reconstructing pitcher stints and removal points...")
    examples = build_hook_examples(pitches)
    logger.info(
        "%d (stint, batter) examples across %d stints (%d removals observed)",
        len(examples), examples["stint_id"].nunique(), int(examples["label"].sum()),
    )

    train_examples = examples[examples["season"].between(*TRAIN_SEASON_RANGE)].reset_index(drop=True)
    val_examples = examples[examples["season"].isin(VAL_SEASONS)].reset_index(drop=True)
    logger.info(
        "Train examples: %d (%d starter, %d reliever), Val examples: %d (%d starter, %d reliever)",
        len(train_examples), int(train_examples["is_starter"].sum()), int((~train_examples["is_starter"]).sum()),
        len(val_examples), int(val_examples["is_starter"].sum()), int((~val_examples["is_starter"]).sum()),
    )

    logger.info(
        "Fitting pooled, starter-only, reliever-only, and hybrid-reliever hazard models, "
        "and comparing removal-point accuracy..."
    )
    starter_model, reliever_model, starter_history, reliever_history, comparison = fit_and_compare_role_specific_models(
        examples, train_examples, val_examples
    )
    logger.info("Removal-point comparison (reliever slot ships the hybrid model, not reliever-only):\n%s", comparison.to_string(index=False))

    predictor = HookModelPredictor(starter_model, reliever_model, starter_history, reliever_history)
    save_predictor(predictor, args.checkpoint)
    logger.info("Saved role-specific predictor to %s", args.checkpoint)


if __name__ == "__main__":
    main()
