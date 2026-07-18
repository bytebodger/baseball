"""A rules-and-probability baserunning-advancement model: given a runner on
a specific base and the current batted-ball outcome, where do they end up?
Deliberately NOT a trained classifier the way the rest of this project's
Phase 4-6 models are (the event model, bullpen availability, the pitcher
hook model) -- every rate here is a direct empirical frequency computed
from real historical outcomes (build_runner_transitions /
compute_league_advancement_rates), adjusted only by a small, hand-set (not
fit) rule when Phase 0's sprint-speed data (src/data/fetch_sprint_speed.py)
is available for the specific runner. That's the scope this module was
asked to stay inside of.

Runner transitions are read directly off the processed pitch table, the
same "compare pre-play state to the next at-bat's post-play state" pattern
src/models/hook_model.py already uses for its own game-state features:
on_1b/on_2b/on_3b hold the actual player_id occupying each base, so a
specific pre-existing runner's fate is determined by finding their own ID
in the following at-bat's base state (still on base -> where; not found ->
scored or put out, disambiguated by the batting team's own score delta
across the play -- see build_runner_transitions for exactly how). A handful
of at-bats with a genuinely ambiguous multi-runner mixed fate (some of
several vanished runners scored, some were put out, and which is which
isn't recoverable from base-occupancy and score alone) are dropped rather
than guessed at, the same right-censoring spirit hook_model.py uses for its
own unresolvable rows.

A runner who vanishes from the bases with no score change is only counted
as a genuine basepath "OUT" if the same half-inning continues afterward
(inning_topbot unchanged). If the half flips, the batting team's 3rd out
just ended the inning and any runner still aboard was left stranded, not
retired on the bases -- a materially different event this module found
itself conflating with real outs on a first pass (roughly half of all
"runner vanished, score unchanged" rows on real data turned out to be
stranded-at-inning-end, not thrown out), so it's checked explicitly and
those rows are dropped like any other unobservable case.

One real granularity limit, inherited from this project's existing outcome
vocabulary (src/data/sequence_dataset.py's OUTCOME_VOCAB, used everywhere
else in this project too): "hit_into_play_out" bundles a sacrifice fly in
with a routine groundout, a double play, a fielder's choice, and more --
this module does not redefine that taxonomy just for itself, so a runner
on 3rd's "hit_into_play_out" scoring rate is a genuine empirical blend of
all of those, not sac-fly-specific. Still a real, honestly-computed rate;
just less granular than a dedicated sac-fly split would give.

Sprint speed enters through exactly one rule (adjust_for_sprint_speed):
identify the empirical distribution's baseline (most common) non-out
outcome and the next-furthest-advanced outcome with any real probability,
then shift a little probability mass between them in proportion to how far
the specific runner's sprint speed sits from league average for that
season. P(OUT) is left untouched -- this is a rule about how often a
runner takes the extra base, not a claim that speed changes how often
they're thrown out attempting it, which is a real effect but a different,
noisier one this deliberately simple module doesn't try to model.

Rates can also be conditioned on the number of outs when the runner's
at-bat began (compute_league_advancement_rates_by_outs,
compare_advancement_rates_by_outs, BaserunningModel's optional `outs`
argument) -- the same "split the same empirical question by a role/context
variable and compare" idea src/models/hook_model.py used for starters vs.
relievers, just applied to a lookup table instead of a trained model. See
compute_league_advancement_rates_by_outs' docstring for a real, easy-to-miss
selection effect this split surfaces: at outs==2, "hit_into_play_out" always
ends the half-inning, so this module's own stranded-runner rule (above)
drops nearly every runner who doesn't score, mechanically flattening the
observed OUT rate toward zero rather than reflecting reduced real risk.

A REAL ASSUMPTION THIS BAKES INTO ANY GAME ENGINE BUILT ON TOP OF THIS
MODULE: measured against real 2024-season-scale data, only 3 of the 15
(start_base, outcome) combinations this module tracks (all three are
`home_run`, which is simply rare enough that none happened to occur with
2 outs already up) have literally zero outs==2 rows and fall back to the
pooled rate outright. The other 12/15 do have real outs==2 samples, often
large ones (hundreds to tens of thousands of rows) -- but in every one of
those 12, the specific OUT share of that sample is collapsed to a fraction
of (usually under 20%, several to exactly 0%) its outs 0/1 rate, for the
structural reason described above, not because two-out runners are truly
almost never thrown out. So while HOME/advance probabilities at outs==2
are usually backed by a real, roughly trustworthy 2-out-specific sample,
the probability of a runner being retired while advancing at outs==2 is,
for effectively all 15 combinations, not a real 2-out measurement at all
-- it is inherited, unlabeled, from 0/1-out behavior, because this module
currently has no way to observe it directly. Any simulator consuming this
module's outs==2 OUT/held rates should treat them as "assumed to resemble
0/1-out behavior," not as independently validated -- until the underlying
right-censoring problem is resolved (e.g. by adding a real basepath-out
event indicator instead of inferring fate from base occupancy and score
alone), this is a genuine, currently-unavoidable limitation, not just a
detail buried in a function's docstring.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.statcast_common import PROCESSED_DATA_DIR, TRAIN_SEASON_RANGE, VAL_SEASONS, read_partitioned

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "baserunning.yaml"
DEFAULT_CHECKPOINT_PATH = Path("checkpoints") / "baserunning_model.pkl"
DEFAULT_SPRINT_SPEED_DIR = PROCESSED_DATA_DIR / "sprint_speed"

# The batted-ball-in-play outcomes a preceding runner actually has to react
# to (see src/data/sequence_dataset.py's OUTCOME_VOCAB) -- not ball/strike/
# foul/walk/strikeout/hit_by_pitch, which either don't put the ball in play
# or (walk/HBP) only force a runner deterministically, no advancement
# decision to model empirically.
BATTED_BALL_OUTCOMES = ["single", "double", "triple", "home_run", "hit_into_play_out"]

# How far "ahead" each possible end state is, for ranking a runner's
# possible fates by how aggressively they advanced.
BASE_ADVANCE_ORDER = {"1B": 1, "2B": 2, "3B": 3, "HOME": 4}


@dataclass
class BaserunningConfig:
    speed_adjustment_sensitivity: float = 0.5

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "BaserunningConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


# ---------------------------------------------------------------------------
# Runner transitions: one row per (pre-existing runner, batted-ball at-bat).
# ---------------------------------------------------------------------------


def _at_bat_state_table(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, at_bat_number): the base/score state as of
    that at-bat's first pitch, that at-bat's own outcome (from its last
    pitch, the one that actually ends it -- same convention
    build_plate_appearances.py uses), and the same base/score state as of
    the *next* at-bat's first pitch -- the true post-play state, since any
    advancement this at-bat's batted ball produced is now reflected there.
    """
    sorted_pitches = pitches.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    first = sorted_pitches.drop_duplicates(subset=["game_pk", "at_bat_number"], keep="first")
    last = sorted_pitches.drop_duplicates(subset=["game_pk", "at_bat_number"], keep="last")[
        ["game_pk", "at_bat_number", "outcome"]
    ]

    at_bats = first[
        ["game_pk", "at_bat_number", "game_date", "season", "inning_topbot", "outs_when_up",
         "home_team", "away_team", "home_score", "away_score", "on_1b", "on_2b", "on_3b"]
    ].merge(last, on=["game_pk", "at_bat_number"])
    at_bats = at_bats.sort_values(["game_pk", "at_bat_number"]).reset_index(drop=True)

    # game_pk/at_bat_number are pandas nullable Int64 on the real processed
    # table -- a .shift()-introduced NA compared against a nullable column
    # produces pd.NA (not True/False), silently poisoning the boolean
    # comparison below. Cast to plain numpy int64 first (safe: guaranteed
    # non-null on is_valid rows) -- same fix src/models/hook_model.py needed.
    for col in ("game_pk", "at_bat_number", "outs_when_up"):
        at_bats[col] = at_bats[col].astype("int64")

    at_bats["next_valid"] = at_bats["game_pk"] == at_bats["game_pk"].shift(-1)
    at_bats["next_home_score"] = at_bats["home_score"].shift(-1)
    at_bats["next_away_score"] = at_bats["away_score"].shift(-1)
    at_bats["next_on_1b"] = at_bats["on_1b"].shift(-1)
    at_bats["next_on_2b"] = at_bats["on_2b"].shift(-1)
    at_bats["next_on_3b"] = at_bats["on_3b"].shift(-1)
    at_bats["next_inning_topbot"] = at_bats["inning_topbot"].shift(-1)
    return at_bats


def build_runner_transitions(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per (pre-existing runner, batted-ball at-bat they were on
    base for): the base they started on, that at-bat's outcome, and where
    they ended up -- a base ("1B"/"2B"/"3B"), "HOME" if they scored, or
    "OUT" if retired. A runner's fate is resolved by finding their own
    player_id in the next at-bat's base state; if they're no longer on any
    base, the batting team's own run total strictly determines scored vs.
    out *except* for a genuinely ambiguous case (several runners vanish the
    same play, and the run total increased by neither zero nor the full
    vanished count) -- those specific runners are dropped, not guessed at.
    The at-bat's own final row (no next at-bat to compare against) is
    dropped the same way.

    A second, easy-to-miss case gets the same treatment: a runner who
    vanishes from the bases with no score change is only a genuine "OUT" if
    the *same* half-inning continues into the next at-bat. If the half
    changes (inning_topbot flips), the batting team's 3rd out just ended
    the inning and every runner still aboard is left stranded -- not
    retired on the bases, just no longer relevant to this half-inning. On
    real data roughly half of "runner vanished, score unchanged" rows are
    actually this (e.g. a runner on 3rd left on base by the inning-ending
    out is not the same event as a runner thrown out trying to score), so
    conflating the two would badly overstate how often a runner is put out
    while advancing. Stranded runners are dropped, the same as an ambiguous
    or unobservable case.

    Also carries the number of outs (0/1/2) when the runner's at-bat began
    through as an "outs" column, so callers can condition rates on out
    count (see compute_league_advancement_rates_by_outs) instead of only
    the pooled, out-count-blind rate compute_league_advancement_rates
    produces.
    """
    at_bats = _at_bat_state_table(pitches)
    at_bats = at_bats[at_bats["outcome"].isin(BATTED_BALL_OUTCOMES) & at_bats["next_valid"]].reset_index(drop=True)

    same_half_inning = (at_bats["inning_topbot"] == at_bats["next_inning_topbot"]).to_numpy()

    is_away_batting = (at_bats["inning_topbot"] == "Top").to_numpy()
    is_batting_team_home = ~is_away_batting

    home_score = at_bats["home_score"].to_numpy(dtype="float64")
    away_score = at_bats["away_score"].to_numpy(dtype="float64")
    next_home_score = at_bats["next_home_score"].to_numpy(dtype="float64")
    next_away_score = at_bats["next_away_score"].to_numpy(dtype="float64")
    score_before = np.where(is_batting_team_home, home_score, away_score)
    score_after = np.where(is_batting_team_home, next_home_score, next_away_score)
    score_delta = score_after - score_before

    on_1b = at_bats["on_1b"].to_numpy()
    on_2b = at_bats["on_2b"].to_numpy()
    on_3b = at_bats["on_3b"].to_numpy()
    next_on_1b = at_bats["next_on_1b"].to_numpy()
    next_on_2b = at_bats["next_on_2b"].to_numpy()
    next_on_3b = at_bats["next_on_3b"].to_numpy()
    outcomes = at_bats["outcome"].to_numpy()
    seasons = at_bats["season"].to_numpy()
    game_dates = at_bats["game_date"].to_numpy()
    outs = at_bats["outs_when_up"].to_numpy()

    rows = []
    for i in range(len(at_bats)):
        pre = [(base, pid) for base, pid in (("1B", on_1b[i]), ("2B", on_2b[i]), ("3B", on_3b[i])) if pd.notna(pid)]
        if not pre:
            continue

        post_base_by_id = {
            pid: base
            for base, pid in (("1B", next_on_1b[i]), ("2B", next_on_2b[i]), ("3B", next_on_3b[i]))
            if pd.notna(pid)
        }
        vanished = [pid for _, pid in pre if pid not in post_base_by_id]
        delta = score_delta[i]

        for base, pid in pre:
            if pid in post_base_by_id:
                end_base = post_base_by_id[pid]
            elif delta == len(vanished):
                end_base = "HOME"
            elif delta == 0 and same_half_inning[i]:
                end_base = "OUT"
            elif delta == 0:
                continue  # stranded when the batting team's 3rd out ended the half-inning -- not a basepath out
            else:
                continue  # ambiguous mixed-fate multi-runner play -- drop

            rows.append(
                {
                    "runner_id": pid, "start_base": base, "outcome": outcomes[i], "end_base": end_base,
                    "season": seasons[i], "game_date": game_dates[i], "outs": outs[i],
                }
            )

    return pd.DataFrame(rows)


def compute_league_advancement_rates(transitions: pd.DataFrame) -> pd.DataFrame:
    """One row per (start_base, outcome, end_base): the empirical
    probability of that transition conditional on (start_base, outcome) --
    rows sharing a (start_base, outcome) sum to 1.0."""
    counts = transitions.groupby(["start_base", "outcome", "end_base"], as_index=False).size()
    totals = counts.groupby(["start_base", "outcome"])["size"].transform("sum")
    counts["probability"] = counts["size"] / totals
    return counts.sort_values(["start_base", "outcome", "probability"], ascending=[True, True, False]).reset_index(drop=True)


def compute_league_advancement_rates_by_outs(transitions: pd.DataFrame) -> pd.DataFrame:
    """The same empirical rates as compute_league_advancement_rates, further
    split by the number of outs (0/1/2) when the runner's at-bat began --
    the same "fit the same thing on a narrower slice and compare" idea
    src/models/hook_model.py used to split starters from relievers, just
    without any actual model-fitting since this whole module is a lookup
    table rather than a trained classifier. Rows sharing an (outs,
    start_base, outcome) sum to 1.0.

    One real selection effect worth knowing before reading this table, and
    the reason the outs==2 column in particular should not be read at face
    value: *any* runner retired on the bases when outs_when_up is already 2
    is, by definition, the batting team's 3rd out, so that play always ends
    the half-inning -- regardless of which outcome (single, double,
    hit_into_play_out, ...) is on the batter's own line. build_runner_
    transitions' stranded-runner rule cannot tell "this runner was thrown
    out attempting to advance" apart from "this runner was simply left
    standing when an unrelated out ended the half-inning" once the half
    changes on the same play -- both look identical (vanished, score
    unchanged, next at-bat belongs to the other team) -- so both are
    dropped rather than guessed at. At outs 0 or 1 an out on the bases is
    at most the 2nd out, the half-inning continues, and the runner's true
    fate (still on base, scored, or out) is fully observable, so those two
    columns are NOT subject to this effect. The outs==2 column, though, is
    structurally biased toward showing fewer OUTs and more scoring/holding
    than truly occurred, for every outcome, because most of what would
    otherwise populate its OUT and held buckets was never observable in
    the first place -- an artifact of what this box-score-level data can
    resolve, not evidence that two-out runners are actually thrown out
    less. hit_into_play_out is the most extreme case (outs_when_up==2 plus
    the batter's own out makes the half-inning end on essentially every
    such play, not just the on-basepaths-out ones), but the same directional
    bias applies to every (start_base, outcome) pair's outs==2 row. See
    main()'s comparison logging for how this actually shows up in the real
    numbers, and prefer the outs 0-vs-1 comparison for anything meant to
    reflect real behavior rather than this measurement limitation.
    """
    counts = transitions.groupby(["outs", "start_base", "outcome", "end_base"], as_index=False).size()
    totals = counts.groupby(["outs", "start_base", "outcome"])["size"].transform("sum")
    counts["probability"] = counts["size"] / totals
    return counts.sort_values(
        ["outs", "start_base", "outcome", "probability"], ascending=[True, True, True, False]
    ).reset_index(drop=True)


def compare_advancement_rates_by_outs(rates_by_outs: pd.DataFrame) -> pd.DataFrame:
    """Reshapes compute_league_advancement_rates_by_outs' long table into
    one row per (start_base, outcome, end_base) with a probability column
    per out count (0/1/2, NaN if that combination was never observed) plus
    max_spread -- the largest gap between any two observed out counts --
    so the biggest out-count-driven differences sort to the top."""
    wide = rates_by_outs.pivot_table(
        index=["start_base", "outcome", "end_base"], columns="outs", values="probability"
    )
    wide.columns = [f"outs_{c}" for c in wide.columns]
    wide["max_spread"] = wide.max(axis=1) - wide.min(axis=1)
    return wide.sort_values("max_spread", ascending=False).reset_index()


# ---------------------------------------------------------------------------
# Sprint speed (Phase 0): per-player lookup with a nearest-season fallback,
# same fallback spirit as src/data/park_factors.py's ParkFactorEmbedding.
# ---------------------------------------------------------------------------


@dataclass
class SprintSpeedHistory:
    speeds_by_player: dict[int, dict[int, float]]  # player_id -> {season: sprint_speed}
    league_avg_by_season: dict[int, float]
    overall_league_avg: float


def build_sprint_speed_history(sprint_speed_table: pd.DataFrame) -> SprintSpeedHistory:
    speeds_by_player: dict[int, dict[int, float]] = {}
    for player_id, group in sprint_speed_table.groupby("batter_id"):
        speeds_by_player[player_id] = dict(zip(group["season"], group["sprint_speed"]))
    league_avg_by_season = sprint_speed_table.groupby("season")["sprint_speed"].mean().to_dict()
    overall_league_avg = float(sprint_speed_table["sprint_speed"].mean())
    return SprintSpeedHistory(speeds_by_player, league_avg_by_season, overall_league_avg)


def sprint_speed_for(history: SprintSpeedHistory, player_id: int, season: int) -> float | None:
    """That player's sprint speed in `season` if on record; otherwise their
    nearest other season on record. None if this player has no sprint-speed
    data at all -- callers fall back to the pure league rate, unadjusted."""
    seasons = history.speeds_by_player.get(player_id)
    if not seasons:
        return None
    if season in seasons:
        return seasons[season]
    nearest = min(seasons, key=lambda s: abs(s - season))
    return seasons[nearest]


def league_avg_sprint_speed_for(history: SprintSpeedHistory, season: int) -> float:
    return history.league_avg_by_season.get(season, history.overall_league_avg)


def adjust_for_sprint_speed(
    distribution: dict[str, float], runner_sprint_speed: float, league_avg_sprint_speed: float, sensitivity: float
) -> dict[str, float]:
    """Shifts probability mass between the empirical distribution's
    baseline (most common) non-out outcome and the next-furthest-advanced
    outcome with nonzero probability, in proportion to how far
    `runner_sprint_speed` sits from `league_avg_sprint_speed` -- the only
    way speed enters this model (see module docstring for why P(OUT) is
    deliberately left untouched). A distribution with fewer than two
    non-out outcomes (nothing to shift between) is returned unchanged.
    """
    non_out = {k: v for k, v in distribution.items() if k != "OUT" and v > 0}
    if len(non_out) < 2:
        return dict(distribution)

    baseline = max(non_out, key=non_out.get)
    candidates = [k for k in non_out if BASE_ADVANCE_ORDER[k] > BASE_ADVANCE_ORDER[baseline]]
    if not candidates:
        return dict(distribution)
    aggressive = min(candidates, key=lambda k: BASE_ADVANCE_ORDER[k])

    relative_speed_diff = (runner_sprint_speed - league_avg_sprint_speed) / league_avg_sprint_speed
    shift = relative_speed_diff * sensitivity * distribution[baseline]
    shift = float(np.clip(shift, -distribution[aggressive], distribution[baseline]))

    result = dict(distribution)
    result[baseline] = distribution[baseline] - shift
    result[aggressive] = distribution[aggressive] + shift
    return result


# ---------------------------------------------------------------------------
# Packaged model + persistence.
# ---------------------------------------------------------------------------


@dataclass
class BaserunningModel:
    rates: pd.DataFrame  # compute_league_advancement_rates' output
    config: BaserunningConfig
    sprint_speed_history: SprintSpeedHistory | None = None
    rates_by_outs: pd.DataFrame | None = None  # compute_league_advancement_rates_by_outs' output

    def __post_init__(self) -> None:
        """Pre-indexes rates/rates_by_outs into plain dicts once, here at
        construction time -- not per league_distribution call. For a
        checkpoint loaded via load_model, "once" really does mean once:
        pickling a plain object (no custom __reduce__/__setstate__, which
        this class doesn't define) captures __dict__ as it stood after this
        method already ran, and unpickling restores that __dict__ directly
        without calling __init__/__post_init__ again -- so a loaded
        checkpoint gets these indexes for free, not rebuilt on every
        process that loads it.

        This exists because profiling a real batched game_engine.py
        simulation run found league_distribution's pandas boolean-mask
        filter (further slowed by the Arrow-string dtype backing
        start_base/outcome/end_base -- see this method's own cost, mostly
        pyarrow compute-kernel comparisons and DataFrame reindexing, not
        the filtering logic itself) was ~60% of total simulation wall
        time. Both rates/rates_by_outs are small, static tables that never
        change after this model is built -- a dict keyed on exactly what
        every real query already asks for is a strictly better fit here
        than any kind of "batch" API (see league_distribution's own note on
        why that framing doesn't apply to this module the way it does to
        Phase 5/6's trained classifiers).

        Keys are cast to plain `str`/`int` explicitly (not left as
        whatever pandas' Arrow-string/nullable-int groupby keys happen to
        be) so a lookup with an ordinary Python str/int from a caller is
        guaranteed to hash and compare the same way, not just "usually
        works out because numpy/Arrow scalars happen to compare equal to
        their Python counterparts."
        """
        self._pooled_index: dict[tuple[str, str], dict[str, float]] = {
            (str(start_base), str(outcome)): dict(zip(group["end_base"].astype(str), group["probability"]))
            for (start_base, outcome), group in self.rates.groupby(["start_base", "outcome"])
        }
        self._outs_index: dict[tuple[int, str, str], dict[str, float]] | None = None
        if self.rates_by_outs is not None:
            self._outs_index = {
                (int(outs), str(start_base), str(outcome)): dict(zip(group["end_base"].astype(str), group["probability"]))
                for (outs, start_base, outcome), group in self.rates_by_outs.groupby(["outs", "start_base", "outcome"])
            }

    def league_distribution(self, start_base: str, outcome: str, outs: int | None = None) -> dict[str, float]:
        """The league-wide empirical distribution for this (start_base,
        outcome) pair -- {} if it was never observed. If `outs` is given
        and this model has an outs-split table with a nonzero-count slice
        for (outs, start_base, outcome), uses that narrower, out-count-
        conditioned distribution instead of the out-count-blind pooled one;
        otherwise falls back to the pooled distribution (either because no
        outs-split table was built, or that specific slice was never
        observed in real data). A single dict lookup against the index
        __post_init__ built, not a pandas filter -- see that method's own
        docstring for why. Returns a fresh dict each call (a shallow copy
        of the cached one), matching this method's original observable
        contract: safe for a caller to mutate without corrupting this
        model's own cached state."""
        if outs is not None and self._outs_index is not None:
            distribution = self._outs_index.get((outs, start_base, outcome))
            if distribution:
                return dict(distribution)
        return dict(self._pooled_index.get((start_base, outcome), {}))

    def advancement_distribution(
        self,
        start_base: str,
        outcome: str,
        runner_id: int | None = None,
        season: int | None = None,
        outs: int | None = None,
    ) -> dict[str, float]:
        """The league distribution (out-count-conditioned if `outs` is
        given and observed -- see league_distribution), adjusted for
        `runner_id`'s own sprint speed in `season` if both are given and
        Phase 0 has data for that player -- otherwise the unadjusted
        distribution."""
        distribution = self.league_distribution(start_base, outcome, outs)
        if not distribution or runner_id is None or season is None or self.sprint_speed_history is None:
            return distribution
        speed = sprint_speed_for(self.sprint_speed_history, runner_id, season)
        if speed is None:
            return distribution
        league_avg = league_avg_sprint_speed_for(self.sprint_speed_history, season)
        return adjust_for_sprint_speed(distribution, speed, league_avg, self.config.speed_adjustment_sensitivity)

    def probability(
        self,
        start_base: str,
        outcome: str,
        end_base: str,
        runner_id: int | None = None,
        season: int | None = None,
        outs: int | None = None,
    ) -> float:
        return self.advancement_distribution(start_base, outcome, runner_id, season, outs).get(end_base, 0.0)


def save_model(model: BaserunningModel, path: Path = DEFAULT_CHECKPOINT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path = DEFAULT_CHECKPOINT_PATH) -> BaserunningModel:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the empirical baserunning-advancement model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--pitches-dir", type=Path, default=PROCESSED_DATA_DIR / "pitches")
    parser.add_argument("--sprint-speed-dir", type=Path, default=DEFAULT_SPRINT_SPEED_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--train-season-start", type=int, default=TRAIN_SEASON_RANGE[0],
        help="Overrides the project-wide default train split start (statcast_common.TRAIN_SEASON_RANGE) -- "
        "e.g. for walk-forward retraining at a later season boundary.",
    )
    parser.add_argument(
        "--val-season-end", type=int, default=VAL_SEASONS[-1],
        help="Overrides the project-wide default validation split end (statcast_common.VAL_SEASONS[-1]) -- "
        "pitches through this season (inclusive) are included in the rate table.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config = BaserunningConfig.from_yaml(args.config)

    logger.info("Loading pitches...")
    full_pitches = read_partitioned(args.pitches_dir)
    pitches = full_pitches[full_pitches["is_valid"]].reset_index(drop=True)
    # Same train/val boundary the event, hook, and bullpen-availability
    # models already fit on -- the test range is held out entirely, not
    # just the model's own training loop, so a game_engine.py simulation of
    # a held-out game doesn't leak that game's (or any other test-season
    # game's) own runner-advancement behavior into the rate table it's
    # scored against. Unlike hook_model.py/bullpen_availability.py this
    # table isn't optimized against val_examples (it's a plain frequency
    # table, not a fitted model needing a validation split), so everything
    # through --val-season-end is used directly rather than held back for
    # its own separate check.
    pitches = pitches[pitches["season"].between(args.train_season_start, args.val_season_end)].reset_index(drop=True)
    logger.info(
        "Restricted to seasons %d-%d (%d pitches) -- everything after --val-season-end excluded entirely.",
        args.train_season_start, args.val_season_end, len(pitches),
    )

    logger.info("Building runner-transition table...")
    transitions = build_runner_transitions(pitches)
    logger.info("%d runner transitions", len(transitions))

    rates = compute_league_advancement_rates(transitions)
    logger.info("League advancement rates:\n%s", rates.to_string(index=False))

    rates_by_outs = compute_league_advancement_rates_by_outs(transitions)
    logger.info("League advancement rates by outs:\n%s", rates_by_outs.to_string(index=False))

    comparison = compare_advancement_rates_by_outs(rates_by_outs)
    logger.info(
        "Out-count comparison, largest spreads first (see compute_league_advancement_rates_by_outs' "
        "docstring for why outs==2 + hit_into_play_out + OUT is a selection artifact, not evidence "
        "of reduced risk-taking):\n%s",
        comparison.to_string(index=False),
    )

    sprint_speed_history = None
    if args.sprint_speed_dir.exists():
        logger.info("Loading sprint speed data from %s...", args.sprint_speed_dir)
        sprint_speed_table = read_partitioned(args.sprint_speed_dir)
        sprint_speed_history = build_sprint_speed_history(sprint_speed_table)
        logger.info("Sprint speed data for %d players.", len(sprint_speed_history.speeds_by_player))
    else:
        logger.warning(
            "No sprint speed data found at %s -- the model will use pure league rates for every runner. "
            "Run `python -m src.data.fetch_sprint_speed` to populate it.", args.sprint_speed_dir,
        )

    model = BaserunningModel(rates, config, sprint_speed_history, rates_by_outs)
    save_model(model, args.checkpoint)
    logger.info("Saved baserunning model to %s", args.checkpoint)


if __name__ == "__main__":
    main()
