"""Pitch-by-pitch, plate-appearance-by-plate-appearance game simulator: wires
together every model this project's earlier phases produced into one
end-to-end at-bat loop.

- Phase 4 (src/models/event_model.py, EventModel): at every pitch, predicts a
  distribution over OUTCOME_VOCAB (src/data/sequence_dataset.py) for the
  current pitcher/batter/situation, which is sampled (not argmaxed -- this is
  a stochastic simulator, the same way a real plate appearance's outcome
  isn't the "most likely" one every time) to decide what actually happened on
  that pitch.
- Phase 7 (src/simulation/baserunning.py, BaserunningModel): on a
  plate-appearance-ending batted ball, resolves each pre-existing runner's
  fate by sampling from the empirical advancement distribution for their own
  (start_base, outcome[, outs]) situation, adjusted for their own sprint
  speed when Phase 0 has it.
- Phase 6 (src/models/hook_model.py, HookModelPredictor): after every batter,
  predicts the current pitcher's removal probability and samples whether
  they're actually removed.
- Phase 5 (src/models/bullpen_availability.py, BullpenAvailabilityPredictor):
  on a removal, scores every not-yet-used pitcher in that team's bullpen and
  brings in whichever one it rates most available.

Four real, inherited limitations worth knowing before using this module:

1. **The event model can only be run for (player_id, game_date) pairs already
   in the precomputed embedding cache** (src/data/event_embedding_cache.py) --
   that cache never computes on a miss, by design, and this module doesn't
   work around that (see EmbeddingCache.get's own docstring for why). In
   practice this means every pitcher_id/batter_id passed to simulate_game
   must be a real player who actually appeared in the historical data the
   cache was built from, and `game_date` must be a date they're cached for
   (typically a date they actually played) -- this simulates alternate
   outcomes of a real, already-observed matchup/date, not a fully
   hypothetical future game with players/dates the cache has never seen.
2. **No mid-plate-appearance baserunning events** (stolen bases, pickoffs,
   wild pitches, passed balls, balks) -- OUTCOME_VOCAB itself has no category
   for any of these, so a runner's base only ever changes at a
   plate-appearance boundary (walk/HBP force, or a batted ball resolved via
   Phase 7). This is a limitation of the upstream event model's own label
   space, not something reintroduced here.
3. **Walk/HBP advancement is deterministic force logic** (apply_force_advance),
   not sampled from Phase 7 -- real force-advancement on a walk/HBP isn't a
   baserunning decision at all (a runner forced off their base has no choice
   in the matter), so there's nothing to model empirically there the way a
   batted ball's advancement is a real decision/outcome to predict.
4. **A batted ball resolves each pre-existing runner independently** (one
   Phase 7 draw per occupied base), which can occasionally sample two runners
   onto the same base -- Phase 7's rates are per-runner marginals, not a
   jointly-consistent multi-runner model (see baserunning.py's own module
   docstring on why). resolve_batted_ball breaks such a collision by bumping
   the trailing runner forward one additional base at a time; this tie-break
   rule is not itself empirically measured, just a simple, documented way to
   keep the simulated state physically valid.

The extra-innings rule (2020+) is implemented directly: every half-inning
from the 10th on starts with a runner already on second -- specifically the
batter in the lineup slot immediately before that half-inning's leadoff
batter, per the actual rule's own wording, not a "whoever made the last out"
heuristic (those coincide for an unpinch-hit lineup, but the rule's own text
is about lineup position, so that's what's implemented).

Bullpen availability has no live 26-man-roster/injury concept (see
bullpen_availability.py's own docstring) and this module adds no roster
constraint of its own beyond "don't reuse a pitcher already used in this
game" -- it is the caller's responsibility to pass a bullpen list deep enough
that the game doesn't run out of fresh arms; if it does, the current pitcher
is kept in past what the hook model wanted (logged, not a crash) rather than
this module inventing a phantom reliever.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.event_dataset import SITUATIONAL_CONTINUOUS_FEATURES
from src.data.event_embedding_cache import DEFAULT_CACHE_DIR as DEFAULT_EMBEDDING_CACHE_DIR
from src.data.event_embedding_cache import EmbeddingCache
from src.data.game_dataset import BATTER_APPEARANCES_DIR, GAMES_DIR, PITCHER_APPEARANCES_DIR, load_game_split
from src.data.park_factors import LeagueRatesIndex, ParkFactorConfig, ParkFactorEmbedding, compute_league_rates, compute_park_factors
from src.data.sequence_dataset import MATCHUP_INDEX, OUTCOME_INDEX, OUTCOME_VOCAB
from src.data.statcast_common import PROCESSED_DATA_DIR, RAW_DATA_DIR, TRAIN_SEASON_RANGE, VAL_SEASONS, read_partitioned
from src.device import DEFAULT_DEVICE, resolve_device
from src.models.bullpen_availability import DEFAULT_CHECKPOINT_PATH as DEFAULT_BULLPEN_AVAILABILITY_CHECKPOINT
from src.models.bullpen_availability import (
    BullpenAvailabilityPredictor,
    PitcherWorkloadHistory,
    build_workload_history,
    compute_entry_situations,
    compute_pitch_counts,
)
from src.models.bullpen_availability import (
    CLOSER_RECENCY_FEATURE_NAMES,
    TEAM_SAVE_OPPORTUNITY_FEATURE_NAME,
    WORKLOAD_FEATURE_NAMES,
    closer_recency_features_for,
    workload_features_for,
)
from src.models.bullpen_availability import load_predictor as load_bullpen_predictor
from src.models.event_model import EventModel, EventModelConfig
from src.models.hook_model import DEFAULT_CHECKPOINT_PATH as DEFAULT_HOOK_MODEL_CHECKPOINT
from src.models.hook_model import (
    PITCH_COUNT_MILESTONE_FEATURE_NAMES,
    PITCH_COUNT_MILESTONES,
    HookModelPredictor,
    removal_history_features_for,
)
from src.models.hook_model import load_predictor as load_hook_predictor
from src.simulation.baserunning import DEFAULT_CHECKPOINT_PATH as DEFAULT_BASERUNNING_CHECKPOINT
from src.simulation.baserunning import BASE_ADVANCE_ORDER, BATTED_BALL_OUTCOMES, BaserunningModel
from src.simulation.baserunning import load_model as load_baserunning_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_EVENT_MODEL_CHECKPOINT = Path("checkpoints") / "event_model_full_best.pt"

# Pitches that don't end the plate appearance -- update the count and keep
# throwing. "UNK" (an unmapped/missing outcome -- see statcast_common's
# compute_outcome) is treated the same way: re-thrown, not fabricated into
# either a ball or a strike.
NON_TERMINAL_OUTCOMES = {"ball", "called_strike", "swinging_strike", "foul", "UNK"}
# Safety net only, not a real-baseball limit: a well-calibrated event model
# should reach a terminal outcome (directly, or via the 4-ball/3-strike force
# below) long before this. Guards against a pathological run of fouls/UNK at
# a 2-strike count hanging the simulator.
MAX_PITCHES_PER_PLATE_APPEARANCE = 30

BASES = ("1B", "2B", "3B")


# ---------------------------------------------------------------------------
# Live game state.
# ---------------------------------------------------------------------------


@dataclass
class TeamState:
    lineup: list[int]  # exactly 9 batter_ids, in batting order
    bullpen: list[int]  # candidate reliever pitcher_ids, not including the starter
    starter_id: int
    is_home: bool
    current_pitcher_id: int = field(init=False)
    batting_index: int = 0
    stint_batters_faced: int = 0
    stint_pitch_count: int = 0
    used_pitcher_ids: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if len(self.lineup) != 9:
            raise ValueError(f"lineup must have exactly 9 batters, got {len(self.lineup)}")
        self.current_pitcher_id = self.starter_id
        self.used_pitcher_ids.add(self.starter_id)


@dataclass
class GameState:
    inning: int = 1
    is_top: bool = True
    outs: int = 0
    bases: dict[str, int | None] = field(default_factory=lambda: {"1B": None, "2B": None, "3B": None})
    home_score: int = 0
    away_score: int = 0


@dataclass
class GameResult:
    home_score: int
    away_score: int
    innings_played: int
    winner: str  # "home" or "away"


# ---------------------------------------------------------------------------
# Model bundle: everything simulate_game needs loaded once and reused across
# many games, mirroring this project's existing "load real checkpoints once,
# call a single-example predict method many times" convention (HookModelPredictor,
# BullpenAvailabilityPredictor, BaserunningModel).
# ---------------------------------------------------------------------------


@dataclass
class GameEngineContext:
    event_model: EventModel
    park_factor_embedding: ParkFactorEmbedding
    situational_stats: dict[str, tuple[float, float]]
    league_rates: pd.DataFrame
    league_rates_index: LeagueRatesIndex
    pitcher_cache: EmbeddingCache
    batter_cache: EmbeddingCache
    handedness: dict[str, dict[int, str]]  # {"pitcher": {id: p_throws}, "batter": {id: stand}}
    hook_predictor: HookModelPredictor
    bullpen_predictor: BullpenAvailabilityPredictor
    workload_history: PitcherWorkloadHistory
    baserunning_model: BaserunningModel
    device: torch.device


def build_handedness_lookup(pitches: pd.DataFrame) -> dict[str, dict[int, str]]:
    """Every pitcher_id's most-recorded p_throws and every batter_id's
    most-recorded stand -- there's no dedicated handedness table anywhere in
    this project (see game_engine research), only per-pitch stand/p_throws
    on the processed pitch table itself, so a live simulator has to derive
    its own lookup from it. "Most recorded" rather than "most recent":
    handedness essentially never changes within a career (switch-hitters
    aside, and Statcast records their actual stand per plate appearance
    anyway, so the mode still reflects real usage), so there's no leakage
    concern in using full history here.
    """
    pitcher_hand = (
        pitches.dropna(subset=["p_throws"]).groupby("pitcher_id")["p_throws"].agg(lambda s: s.value_counts().idxmax())
    )
    batter_hand = pitches.dropna(subset=["stand"]).groupby("batter_id")["stand"].agg(lambda s: s.value_counts().idxmax())
    return {"pitcher": pitcher_hand.to_dict(), "batter": batter_hand.to_dict()}


def build_game_engine_context(
    pitches_dir: Path = PROCESSED_DATA_DIR / "pitches",
    embedding_cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
    event_model_checkpoint: Path = DEFAULT_EVENT_MODEL_CHECKPOINT,
    hook_model_checkpoint: Path = DEFAULT_HOOK_MODEL_CHECKPOINT,
    bullpen_availability_checkpoint: Path = DEFAULT_BULLPEN_AVAILABILITY_CHECKPOINT,
    baserunning_checkpoint: Path = DEFAULT_BASERUNNING_CHECKPOINT,
    raw_dir: Path = RAW_DATA_DIR,
    games_dir: Path = GAMES_DIR,
    pitcher_appearances_dir: Path = PITCHER_APPEARANCES_DIR,
    batter_appearances_dir: Path = BATTER_APPEARANCES_DIR,
    device: str = DEFAULT_DEVICE,
) -> GameEngineContext:
    """One-time setup (real checkpoints + real historical data, several
    seconds -- dominated by rebuilding the park-factor embedding and the
    bullpen workload history): loads every model simulate_game needs and
    bundles them into a GameEngineContext meant to be built once and reused
    across many simulate_game calls, not rebuilt per game."""
    resolved_device = resolve_device(device)

    logger.info("Loading pitches from %s", pitches_dir)
    full_pitches = read_partitioned(pitches_dir)
    valid_pitches = full_pitches[full_pitches["is_valid"]].reset_index(drop=True)

    # The event model's park-factor embedding table and league-rate lookup
    # must be rebuilt from *exactly* the pitches train_event_model.py used
    # (train+val seasons only) -- any other season range produces a
    # different number of (park_id, season) rows, and load_state_dict raises
    # a shape mismatch against the trained embedding table already baked
    # into the checkpoint.
    event_model_pitches = valid_pitches[valid_pitches["season"].between(TRAIN_SEASON_RANGE[0], VAL_SEASONS[-1])].reset_index(
        drop=True
    )

    logger.info("Loading event model checkpoint from %s", event_model_checkpoint)
    ckpt = torch.load(event_model_checkpoint, map_location="cpu", weights_only=False)
    model_config = EventModelConfig(**ckpt["model_config"])
    park_factor_config = ParkFactorConfig(**ckpt["park_factor_config"])
    situational_stats = ckpt["situational_stats"]

    park_factors = compute_park_factors(event_model_pitches, rolling_years=park_factor_config.rolling_years)
    park_factor_embedding = ParkFactorEmbedding(park_factor_config, park_factors)
    league_rates = compute_league_rates(event_model_pitches, rolling_years=park_factor_config.rolling_years)
    league_rates_index = LeagueRatesIndex(league_rates)

    event_model = EventModel(model_config, park_factor_embedding)
    event_model.load_state_dict(ckpt["model_state_dict"])
    event_model.to(resolved_device)
    event_model.eval()

    pitcher_cache = EmbeddingCache(embedding_cache_dir, "pitcher")
    batter_cache = EmbeddingCache(embedding_cache_dir, "batter")
    handedness = build_handedness_lookup(full_pitches)

    logger.info("Loading hook model predictor from %s", hook_model_checkpoint)
    hook_predictor = load_hook_predictor(hook_model_checkpoint)

    logger.info("Loading bullpen availability predictor from %s", bullpen_availability_checkpoint)
    bullpen_predictor = load_bullpen_predictor(bullpen_availability_checkpoint)

    logger.info("Building pitcher workload history...")
    pitch_counts = compute_pitch_counts(valid_pitches)
    entry_situations = compute_entry_situations(valid_pitches)
    _, _, pitcher_appearances, _ = load_game_split(
        raw_dir=raw_dir,
        games_dir=games_dir,
        pitcher_appearances_dir=pitcher_appearances_dir,
        batter_appearances_dir=batter_appearances_dir,
    )
    workload_history = build_workload_history(pitcher_appearances, pitch_counts, entry_situations)

    logger.info("Loading baserunning model from %s", baserunning_checkpoint)
    baserunning_model = load_baserunning_model(baserunning_checkpoint)

    return GameEngineContext(
        event_model=event_model,
        park_factor_embedding=park_factor_embedding,
        situational_stats=situational_stats,
        league_rates=league_rates,
        league_rates_index=league_rates_index,
        pitcher_cache=pitcher_cache,
        batter_cache=batter_cache,
        handedness=handedness,
        hook_predictor=hook_predictor,
        bullpen_predictor=bullpen_predictor,
        workload_history=workload_history,
        baserunning_model=baserunning_model,
        device=resolved_device,
    )


# ---------------------------------------------------------------------------
# Sampling.
# ---------------------------------------------------------------------------


def sample_categorical(distribution: dict[str, float], rng: np.random.Generator) -> str:
    labels = list(distribution.keys())
    weights = np.array(list(distribution.values()), dtype="float64")
    total = weights.sum()
    if total <= 0:
        raise ValueError(f"Cannot sample from a distribution with non-positive total probability: {distribution!r}")
    return labels[rng.choice(len(labels), p=weights / total)]


# ---------------------------------------------------------------------------
# Phase 4: per-pitch outcome distribution + the pitch-by-pitch plate
# appearance loop built on top of it.
# ---------------------------------------------------------------------------


def event_outcome_distribution(
    context: GameEngineContext,
    pitcher_id: int,
    batter_id: int,
    game_date,
    season: int,
    park_id: str,
    balls: int,
    strikes: int,
    outs_when_up: int,
    score_diff: int,
    inning: int,
    times_through_order: int,
    bases: dict[str, int | None],
) -> dict[str, float]:
    """One EventModel forward pass for one live pitch -- builds the same
    shaped input EventBatchCollator produces for training, batch size 1."""
    device = context.device
    pitcher_embedding = context.pitcher_cache.get(pitcher_id, game_date).unsqueeze(0).to(device)
    batter_embedding = context.batter_cache.get(batter_id, game_date).unsqueeze(0).to(device)

    raw_situational = {
        "balls": balls,
        "strikes": strikes,
        "outs_when_up": outs_when_up,
        "score_diff": score_diff,
        "inning": inning,
        "times_through_order": times_through_order,
    }
    situational = torch.tensor(
        [[(raw_situational[col] - context.situational_stats[col][0]) / context.situational_stats[col][1] for col in SITUATIONAL_CONTINUOUS_FEATURES]],
        dtype=torch.float32,
        device=device,
    )
    # BASE_STATE_COLUMNS ("on_1b", "on_2b", "on_3b") is exactly BASES
    # ("1B", "2B", "3B") order, so building straight from BASES here matches
    # EventBatchCollator's own column order without needing the name mapping.
    base_state = torch.tensor(
        [[float(bases[label] is not None) for label in BASES]],
        dtype=torch.float32,
        device=device,
    )

    pitcher_hand = context.handedness["pitcher"].get(pitcher_id)
    batter_hand = context.handedness["batter"].get(batter_id)
    matchup_key = f"{batter_hand}_{pitcher_hand}" if pitcher_hand and batter_hand else "UNK"
    matchup_index = torch.tensor([MATCHUP_INDEX.get(matchup_key, MATCHUP_INDEX["UNK"])], dtype=torch.long, device=device)

    park_index = torch.tensor([context.park_factor_embedding.index_for(park_id, season)], dtype=torch.long, device=device)

    hr_rate, runs_rate = context.league_rates_index.for_season(season)
    league_rates_tensor = torch.tensor([[hr_rate, runs_rate]], dtype=torch.float32, device=device)

    batch_context = torch.cat([situational, base_state, league_rates_tensor], dim=-1)
    batch = {
        "pitcher_embedding": pitcher_embedding,
        "batter_embedding": batter_embedding,
        "context": batch_context,
        "matchup_index": matchup_index,
        "park_index": park_index,
    }
    with torch.no_grad():
        logits = context.event_model(batch)
    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    return dict(zip(OUTCOME_VOCAB, probs.tolist()))


def simulate_plate_appearance(
    context: GameEngineContext,
    rng: np.random.Generator,
    pitcher_id: int,
    batter_id: int,
    game_date,
    season: int,
    park_id: str,
    outs_when_up: int,
    score_diff: int,
    inning: int,
    times_through_order: int,
    bases: dict[str, int | None],
) -> tuple[str, int]:
    """Throws pitches (each one an EventModel call + sample) until a
    plate-appearance-ending outcome is reached. Returns (outcome,
    pitches_thrown). balls/strikes are tracked locally as a live count *and*
    a safety net: walk/strikeout are themselves directly-sampled terminal
    categories in OUTCOME_VOCAB (the real historical label on the actual
    4th-ball/3rd-strike pitch is "walk"/"strikeout", not one more
    "ball"/strike category), so in the common case the model samples the
    terminal category itself right on count. The explicit balls>=4 /
    strikes>=3 forcing below only fires if the model instead samples one
    more non-terminal ball/strike despite the count already being full --
    real-baseball-illegal, so it's forced to the terminal outcome rather
    than let the count run past 4-0 or 3-2.
    """
    balls = 0
    strikes = 0
    for pitch_number in range(1, MAX_PITCHES_PER_PLATE_APPEARANCE + 1):
        distribution = event_outcome_distribution(
            context, pitcher_id, batter_id, game_date, season, park_id,
            balls, strikes, outs_when_up, score_diff, inning, times_through_order, bases,
        )
        outcome = sample_categorical(distribution, rng)

        if outcome not in NON_TERMINAL_OUTCOMES:
            return outcome, pitch_number

        if outcome == "ball":
            balls += 1
            if balls >= 4:
                return "walk", pitch_number
        elif outcome == "foul":
            strikes = min(strikes + 1, 2)  # a foul with 2 strikes doesn't add a 3rd
        elif outcome in ("called_strike", "swinging_strike"):
            strikes += 1
            if strikes >= 3:
                return "strikeout", pitch_number
        # "UNK": no count change, pitch simply re-thrown.

    logger.warning(
        "Plate appearance (pitcher_id=%s, batter_id=%s) exceeded %d pitches without a terminal outcome -- "
        "forcing hit_into_play_out.", pitcher_id, batter_id, MAX_PITCHES_PER_PLATE_APPEARANCE,
    )
    return "hit_into_play_out", MAX_PITCHES_PER_PLATE_APPEARANCE


# ---------------------------------------------------------------------------
# Resolving a plate appearance's outcome into runs / outs / base state.
# ---------------------------------------------------------------------------


def apply_force_advance(bases: dict[str, int | None], batter_id: int) -> tuple[int, int, dict[str, int | None]]:
    """Deterministic force-advancement on a walk/HBP -- not a baserunning
    decision (a forced runner has no choice), so nothing here is sampled
    from Phase 7. Standard cascading force: the batter always takes 1st;
    each preceding runner is forced off their base only if every base behind
    them (down through 1st) was already occupied."""
    new_bases = dict(bases)
    runs = 0
    if new_bases["1B"] is not None:
        if new_bases["2B"] is not None:
            if new_bases["3B"] is not None:
                runs += 1
            new_bases["3B"] = new_bases["2B"]
        new_bases["2B"] = new_bases["1B"]
    new_bases["1B"] = batter_id
    return runs, 0, new_bases


def resolve_batted_ball(
    baserunning_model: BaserunningModel,
    rng: np.random.Generator,
    bases: dict[str, int | None],
    outcome: str,
    outs_when_up: int,
    season: int,
) -> tuple[int, int, dict[str, int | None]]:
    """Resolves every pre-existing runner's fate on a batted-ball outcome
    (see BATTED_BALL_OUTCOMES), most-advanced runner first, sampling each
    one's destination from Phase 7's empirical advancement distribution for
    their own (start_base, outcome, outs) situation and that runner's own
    sprint speed if Phase 0 has it. Does not place the batter -- see
    place_batter. Returns (runs_scored, runners_put_out, new_bases)."""
    new_bases: dict[str, int | None] = {"1B": None, "2B": None, "3B": None}
    claimed: set[str] = set()
    runs = 0
    outs = 0

    for start_base in ("3B", "2B", "1B"):
        runner_id = bases.get(start_base)
        if runner_id is None:
            continue

        distribution = baserunning_model.advancement_distribution(
            start_base, outcome, runner_id=runner_id, season=season, outs=outs_when_up
        )
        if not distribution:
            # No empirical data at all for this (start_base, outcome) pair
            # (see BaserunningModel.league_distribution) -- hold the runner
            # in place rather than inventing a probability.
            end_base = start_base
        else:
            end_base = sample_categorical(distribution, rng)

        if end_base == "OUT":
            outs += 1
            continue
        if end_base != "HOME":
            # Two runners can't occupy the same base. Phase 7's rates are
            # independent per-runner marginals, not a jointly-consistent
            # multi-runner model (see baserunning.py's own module docstring),
            # so a collision is possible here -- resolved by bumping the
            # trailing runner (this loop processes lead runners first, so
            # `claimed` only ever holds more-advanced runners' spots) one
            # additional base at a time. Not itself an empirically measured
            # rule, just a simple, documented way to keep the simulated
            # state physically valid.
            while end_base in claimed:
                idx = BASE_ADVANCE_ORDER[end_base]
                end_base = {1: "2B", 2: "3B", 3: "HOME"}[idx]
                if end_base == "HOME":
                    break
        if end_base == "HOME":
            runs += 1
        else:
            claimed.add(end_base)
            new_bases[end_base] = runner_id

    return runs, outs, new_bases


def place_batter(new_bases: dict[str, int | None], outcome: str, batter_id: int) -> tuple[int, int]:
    """Where the batter themself ends up on a batted-ball outcome --
    deterministic given the outcome category, not sampled: single/double/
    triple send the batter to 1st/2nd/3rd, home_run scores them directly,
    and hit_into_play_out is, by this outcome bucket's own definition (see
    baserunning.py's module docstring on its known fielder's-choice/error
    blend), always a batter out. Returns (runs_from_batter, batter_is_out).
    Mutates new_bases in place for the non-out cases."""
    if outcome == "home_run":
        return 1, 0
    if outcome == "hit_into_play_out":
        return 0, 1
    base = {"single": "1B", "double": "2B", "triple": "3B"}[outcome]
    new_bases[base] = batter_id
    return 0, 0


def apply_outcome(
    baserunning_model: BaserunningModel,
    rng: np.random.Generator,
    bases: dict[str, int | None],
    outcome: str,
    batter_id: int,
    outs_when_up: int,
    season: int,
) -> tuple[int, int, dict[str, int | None]]:
    """Dispatches a plate appearance's terminal outcome to the right
    resolution rule. Returns (runs_scored, new_outs, new_bases)."""
    if outcome == "strikeout":
        return 0, 1, dict(bases)
    if outcome in ("walk", "hit_by_pitch"):
        return apply_force_advance(bases, batter_id)
    if outcome in BATTED_BALL_OUTCOMES:
        runs, additional_outs, new_bases = resolve_batted_ball(baserunning_model, rng, bases, outcome, outs_when_up, season)
        batter_runs, batter_out = place_batter(new_bases, outcome, batter_id)
        return runs + batter_runs, additional_outs + batter_out, new_bases
    raise ValueError(f"Unhandled plate-appearance-ending outcome: {outcome!r}")


# ---------------------------------------------------------------------------
# Phase 6 + Phase 5: hook decision and reliever selection.
# ---------------------------------------------------------------------------


def select_replacement_pitcher(context: GameEngineContext, pitching_team: TeamState, game_date) -> int | None:
    """Scores every not-yet-used pitcher in this team's bullpen and returns
    whichever one Phase 5 rates most available -- None if every candidate
    has already been used this game (see module docstring: this module adds
    no roster constraint of its own beyond "don't reuse a pitcher")."""
    candidates = [pid for pid in pitching_team.bullpen if pid not in pitching_team.used_pitcher_ids]
    if not candidates:
        return None
    scored = [
        (context.bullpen_predictor.predict_proba(context.workload_history, pid, game_date), pid) for pid in candidates
    ]
    return max(scored, key=lambda pair: pair[0])[1]


def maybe_replace_pitcher(
    context: GameEngineContext,
    rng: np.random.Generator,
    pitching_team: TeamState,
    state: GameState,
    game_date,
    times_through_order: int,
    verbose: bool = False,
) -> None:
    """After a batter: samples Phase 6's removal decision from the pitcher's
    current stint state and the game state as it now stands (post-plate-
    appearance, post-half-inning-transition if one just happened -- the same
    "next at-bat's first pitch" convention build_hook_examples trains
    against), and on a removal, brings in a Phase 5-selected replacement.
    Mutates pitching_team in place. A removal with no available bullpen
    candidates left is logged and the current pitcher stays in -- see module
    docstring."""
    is_starter = pitching_team.current_pitcher_id == pitching_team.starter_id
    pitcher_team_score = state.home_score if pitching_team.is_home else state.away_score
    opponent_score = state.away_score if pitching_team.is_home else state.home_score
    run_differential = float(pitcher_team_score - opponent_score)
    runner_on_base = any(v is not None for v in state.bases.values())

    removal_probability = context.hook_predictor.predict_proba(
        pitching_team.current_pitcher_id,
        game_date,
        is_starter,
        pitching_team.stint_batters_faced,
        pitching_team.stint_pitch_count,
        run_differential,
        runner_on_base,
        times_through_order=times_through_order,
    )
    if rng.random() >= removal_probability:
        return

    replacement_id = select_replacement_pitcher(context, pitching_team, game_date)
    if replacement_id is None:
        logger.warning(
            "Hook model called for removing pitcher_id=%s but no unused bullpen arm remains -- keeping them in.",
            pitching_team.current_pitcher_id,
        )
        return

    if verbose:
        logger.info(
            "  [pitching change] %s replaces pitcher_id=%s (%d batters faced, %d pitches)",
            replacement_id, pitching_team.current_pitcher_id, pitching_team.stint_batters_faced, pitching_team.stint_pitch_count,
        )

    pitching_team.current_pitcher_id = replacement_id
    pitching_team.used_pitcher_ids.add(replacement_id)
    pitching_team.stint_batters_faced = 0
    pitching_team.stint_pitch_count = 0


# ---------------------------------------------------------------------------
# The game loop.
# ---------------------------------------------------------------------------


def _format_bases(bases: dict[str, int | None]) -> str:
    occupied = [f"{base}={pid}" for base, pid in bases.items() if pid is not None]
    return ",".join(occupied) if occupied else "empty"


def simulate_game(
    home_starter: int,
    away_starter: int,
    home_lineup: list[int],
    away_lineup: list[int],
    home_bullpen: list[int],
    away_bullpen: list[int],
    park_id: str,
    game_date,
    context: GameEngineContext,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> GameResult:
    """Simulates one full game, pitch by pitch, using the models bundled in
    `context` (see build_game_engine_context). `context` and `rng` are the
    only parameters beyond what defines the game itself: `context` bundles
    every loaded model/history this function needs (built once, reused
    across many games -- see its own docstring for why); `rng` defaults to a
    fresh np.random.default_rng() and exists so a caller can pass their own
    for reproducible simulations. `verbose=True` logs a play-by-play line
    (logger.info) for every plate appearance -- half-inning/outs/score
    before the play, pitcher vs. batter, the sampled outcome and how many
    pitches it took, and the resulting base state -- plus half-inning
    transitions, extra-innings ghost-runner placement, pitching changes, and
    the final result. Off by default since a batch of many simulated games
    (see the bullpen-exhaustion study) would otherwise flood the log; turn
    it on for a single game you actually want to read through.

    Tracks, and updates at every pitch or plate appearance as appropriate:
    inning, top/bottom, outs, score, baserunners, each team's batting-order
    position, each side's current pitcher, that pitcher's own stint pitch
    count and batters faced, and times-through-the-order for the specific
    (pitcher, batter) matchup currently at the plate (the same
    n_thruorder_pitcher-derived quantity src/data/statcast_common.py computes
    from real games -- how many times *this* pitcher has already faced *this*
    batter this game, not a lineup-wide cycle count).

    Regulation ends after 9 innings once the score is no longer tied (or
    immediately, without a bottom 9th, if the home team already leads after
    the top of the 9th or later) -- or continues into extra innings, each
    starting with a runner already on 2nd (the 2020+ rule; see module
    docstring), until a winner is decided. A go-ahead run for the home team
    in the bottom of the 9th or later ends the game immediately (a walk-off),
    even mid-plate-appearance-sequence -- it doesn't wait for three outs.
    """
    rng = rng or np.random.default_rng()
    season = pd.Timestamp(game_date).year

    home = TeamState(lineup=list(home_lineup), bullpen=list(home_bullpen), starter_id=home_starter, is_home=True)
    away = TeamState(lineup=list(away_lineup), bullpen=list(away_bullpen), starter_id=away_starter, is_home=False)

    state = GameState()
    times_faced: dict[tuple[int, int], int] = {}

    if verbose:
        logger.info("=== Game start: away_starter=%s vs home_starter=%s at %s (%s) ===", away_starter, home_starter, park_id, game_date)
        logger.info("--- Top 1 ---")

    while True:
        batting_team, pitching_team = (away, home) if state.is_top else (home, away)

        batter_id = batting_team.lineup[batting_team.batting_index % 9]
        pitcher_id = pitching_team.current_pitcher_id
        matchup_key = (pitcher_id, batter_id)
        times_through_order = times_faced.get(matchup_key, 0)

        batting_score = state.away_score if state.is_top else state.home_score
        fielding_score = state.home_score if state.is_top else state.away_score
        score_diff = batting_score - fielding_score

        outs_before = state.outs
        bases_before = dict(state.bases)

        outcome, pitches_thrown = simulate_plate_appearance(
            context, rng, pitcher_id, batter_id, game_date, season, park_id,
            state.outs, score_diff, state.inning, times_through_order, state.bases,
        )

        pitching_team.stint_pitch_count += pitches_thrown
        pitching_team.stint_batters_faced += 1
        times_faced[matchup_key] = times_through_order + 1

        runs, new_outs, new_bases = apply_outcome(
            context.baserunning_model, rng, state.bases, outcome, batter_id, state.outs, season
        )

        if state.is_top:
            state.away_score += runs
        else:
            state.home_score += runs
        state.outs = min(state.outs + new_outs, 3)
        if state.outs < 3:
            state.bases = new_bases

        if verbose:
            logger.info(
                "  [%d out] pitcher=%s batter=%s -> %s (%dp) | bases %s -> %s | +%d run(s) | score away %d-%d home",
                outs_before, pitcher_id, batter_id, outcome, pitches_thrown,
                _format_bases(bases_before), _format_bases(new_bases),
                runs, state.away_score, state.home_score,
            )

        batting_team.batting_index += 1

        # Walk-off: the go-ahead run in the bottom of the 9th or later ends
        # the game immediately, regardless of the out count on this same play.
        if not state.is_top and state.inning >= 9 and state.home_score > state.away_score:
            if verbose:
                logger.info("=== Walk-off! Final: away %d, home %d ===", state.away_score, state.home_score)
            return _build_result(state)

        if state.outs >= 3:
            if state.is_top and state.inning >= 9 and state.home_score > state.away_score:
                # Home team already leads after the top half -- no bottom
                # half needed.
                if verbose:
                    logger.info(
                        "=== Home leads after the top of inning %d -- no bottom half needed. Final: away %d, home %d ===",
                        state.inning, state.away_score, state.home_score,
                    )
                return _build_result(state)
            if not state.is_top and state.inning >= 9 and state.home_score != state.away_score:
                if verbose:
                    logger.info("=== Final: away %d, home %d ===", state.away_score, state.home_score)
                return _build_result(state)

            if not state.is_top:
                state.inning += 1
            state.is_top = not state.is_top
            state.outs = 0
            next_batting_team = away if state.is_top else home
            if state.inning > 9:
                # 2020+ extra-innings rule: the runner placed on 2nd is the
                # batter in the lineup slot immediately before this
                # half-inning's own leadoff batter -- the rule's own
                # wording, not a "last batter put out" heuristic (they
                # coincide for an unpinch-hit lineup, but this is what the
                # rule actually specifies).
                ghost_runner_id = next_batting_team.lineup[(next_batting_team.batting_index - 1) % 9]
                state.bases = {"1B": None, "2B": ghost_runner_id, "3B": None}
                if verbose:
                    logger.info(
                        "--- %s %d (extra innings: ghost runner %s placed on 2B) --- score away %d-%d home",
                        "Top" if state.is_top else "Bot", state.inning, ghost_runner_id, state.away_score, state.home_score,
                    )
            else:
                state.bases = {"1B": None, "2B": None, "3B": None}
                if verbose:
                    logger.info(
                        "--- %s %d --- score away %d-%d home",
                        "Top" if state.is_top else "Bot", state.inning, state.away_score, state.home_score,
                    )

        # Phase 6/5: hook decision, evaluated using the game state as it now
        # stands -- including any half-inning transition (and ghost runner)
        # just applied above, matching hook_model.py's own "next at-bat's
        # first pitch" convention.
        maybe_replace_pitcher(context, rng, pitching_team, state, game_date, times_through_order, verbose=verbose)


def _build_result(state: GameState) -> GameResult:
    winner = "home" if state.home_score > state.away_score else "away"
    return GameResult(
        home_score=state.home_score, away_score=state.away_score, innings_played=state.inning, winner=winner
    )


# ---------------------------------------------------------------------------
# Batched simulation: many independent replays of the *same* matchup at
# once, vectorized across a batch dimension for the one genuinely GPU-bound,
# high-frequency operation in this whole simulator -- the EventModel's
# per-pitch call. A single game throws roughly 4x as many pitches as it has
# plate appearances, so the event model is called far more often than the
# baserunning/hook/bullpen models combined; batching just that call (a
# batch-of-B forward pass instead of B separate batch-of-1 calls) is where
# essentially all of the available speedup lives, since a batch-of-1
# EventModel call is dominated by fixed Python/dispatch overhead, not GPU
# compute (see the event_model latency figures this project's research
# already measured: ~0.2ms/call, almost all of it not the actual matmuls).
#
# Baserunning (Phase 7), hook (Phase 6), and bullpen (Phase 5) decisions are
# NOT vectorized here -- they stay exactly the single-instance sklearn/pandas
# calls simulate_game already uses (apply_outcome, maybe_replace_pitcher),
# looped in plain Python over whichever games are actually active a given
# round. Two reasons, not just one: (1) they're cheap, non-GPU calls that
# happen once per plate appearance rather than once per pitch, so batching
# them wouldn't move the needle on wall-clock time the way the event model
# does; (2) their real implementations live in Phase 5/6/7's own modules as
# single-instance predict_proba/advancement_distribution APIs -- rewriting
# those into batched tensor operations would be a much larger undertaking
# than this module's own scope, and reusing simulate_game's already-tested
# functions unchanged (rather than reimplementing the same decisions a
# second time against tensors) is what keeps the batched and single-game
# paths from silently drifting apart.
#
# The batch dimension is genuinely a set of *independent* simulations, not
# one simulation replicated: every stochastic decision (event outcome,
# baserunning advancement, hook removal, bullpen selection) draws its own
# sample from `rng` per game, so B replays of an identical matchup diverge
# from the very first pitch, same as running simulate_game B separate times
# with different seeds. Different games in the batch will finish at
# different real-time points (a walk-off in one game doesn't end the whole
# batch) -- a `done` mask keeps finished games frozen while the rest of the
# batch continues, so the loop runs until every simulation is done, bounded
# by whichever one takes longest.
# ---------------------------------------------------------------------------

EMPTY_BASE = -1  # sentinel for "no runner" in the batched bases tensor (real player_ids are always positive)


@dataclass
class BatchTeamState:
    lineup: torch.Tensor  # [B, 9] long
    starter_id: torch.Tensor  # [B] long
    current_pitcher_id: torch.Tensor  # [B] long
    is_home: bool
    batting_index: torch.Tensor  # [B] long
    stint_batters_faced: torch.Tensor  # [B] long
    stint_pitch_count: torch.Tensor  # [B] long
    bullpen: list[list[int]]  # length B -- ragged, kept as plain Python (small, not GPU-bound)
    used_pitcher_ids: list[set[int]]  # length B

    @classmethod
    def create(cls, batch_size: int, lineup: list[int], bullpen: list[int], starter_id: int, is_home: bool) -> "BatchTeamState":
        if len(lineup) != 9:
            raise ValueError(f"lineup must have exactly 9 batters, got {len(lineup)}")
        return cls(
            lineup=torch.tensor([lineup] * batch_size, dtype=torch.long),
            starter_id=torch.full((batch_size,), starter_id, dtype=torch.long),
            current_pitcher_id=torch.full((batch_size,), starter_id, dtype=torch.long),
            is_home=is_home,
            batting_index=torch.zeros(batch_size, dtype=torch.long),
            stint_batters_faced=torch.zeros(batch_size, dtype=torch.long),
            stint_pitch_count=torch.zeros(batch_size, dtype=torch.long),
            bullpen=[list(bullpen) for _ in range(batch_size)],
            used_pitcher_ids=[{starter_id} for _ in range(batch_size)],
        )


@dataclass
class BatchGameState:
    inning: torch.Tensor  # [B] long
    is_top: torch.Tensor  # [B] bool
    outs: torch.Tensor  # [B] long
    bases: torch.Tensor  # [B, 3] long (EMPTY_BASE = empty), columns 1B/2B/3B
    home_score: torch.Tensor  # [B] long
    away_score: torch.Tensor  # [B] long
    done: torch.Tensor  # [B] bool

    @classmethod
    def create(cls, batch_size: int) -> "BatchGameState":
        return cls(
            inning=torch.ones(batch_size, dtype=torch.long),
            is_top=torch.ones(batch_size, dtype=torch.bool),
            outs=torch.zeros(batch_size, dtype=torch.long),
            bases=torch.full((batch_size, 3), EMPTY_BASE, dtype=torch.long),
            home_score=torch.zeros(batch_size, dtype=torch.long),
            away_score=torch.zeros(batch_size, dtype=torch.long),
            done=torch.zeros(batch_size, dtype=torch.bool),
        )


def batched_event_outcome_distribution(
    context: GameEngineContext,
    pitcher_ids: torch.Tensor,
    batter_ids: torch.Tensor,
    game_date,
    season: int,
    park_id: str,
    balls: torch.Tensor,
    strikes: torch.Tensor,
    outs_when_up: torch.Tensor,
    score_diff: torch.Tensor,
    inning: torch.Tensor,
    times_through_order: torch.Tensor,
    bases: torch.Tensor,
) -> torch.Tensor:
    """Batched counterpart to event_outcome_distribution: one EventModel
    forward pass covering every row (one active simulation's current pitch)
    at once. All tensor arguments share the same leading dimension N (the
    number of simulations being asked for a pitch this round -- not
    necessarily the full batch size, since already-finished simulations are
    excluded by the caller). Returns [N, len(OUTCOME_VOCAB)] probabilities
    on context.device.
    """
    n = pitcher_ids.shape[0]
    device = context.device
    game_dates = pd.Series([game_date] * n)
    pitcher_embedding = context.pitcher_cache.get_batch(pd.Series(pitcher_ids.tolist()), game_dates).to(device)
    batter_embedding = context.batter_cache.get_batch(pd.Series(batter_ids.tolist()), game_dates).to(device)

    raw = torch.stack(
        [balls, strikes, outs_when_up, score_diff, inning, times_through_order], dim=1
    ).to(dtype=torch.float32)
    means = torch.tensor([context.situational_stats[c][0] for c in SITUATIONAL_CONTINUOUS_FEATURES])
    stds = torch.tensor([context.situational_stats[c][1] for c in SITUATIONAL_CONTINUOUS_FEATURES])
    situational = ((raw - means) / stds).to(device)

    base_state = (bases != EMPTY_BASE).to(dtype=torch.float32).to(device)

    matchup_indices = []
    for i in range(n):
        pitcher_hand = context.handedness["pitcher"].get(int(pitcher_ids[i]))
        batter_hand = context.handedness["batter"].get(int(batter_ids[i]))
        key = f"{batter_hand}_{pitcher_hand}" if pitcher_hand and batter_hand else "UNK"
        matchup_indices.append(MATCHUP_INDEX.get(key, MATCHUP_INDEX["UNK"]))
    matchup_index = torch.tensor(matchup_indices, dtype=torch.long, device=device)

    park_index = torch.full((n,), context.park_factor_embedding.index_for(park_id, season), dtype=torch.long, device=device)

    hr_rate, runs_rate = context.league_rates_index.for_season(season)
    league_rates_tensor = torch.tensor([[hr_rate, runs_rate]], dtype=torch.float32, device=device).expand(n, -1)

    batch = {
        "pitcher_embedding": pitcher_embedding,
        "batter_embedding": batter_embedding,
        "context": torch.cat([situational, base_state, league_rates_tensor], dim=-1),
        "matchup_index": matchup_index,
        "park_index": park_index,
    }
    with torch.no_grad():
        logits = context.event_model(batch)
    return F.softmax(logits, dim=-1)


def batched_sample_outcome_indices(probs: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """One categorical sample per row of `probs`, natively vectorized via
    torch.multinomial (runs on whatever device `probs` is already on)
    rather than looping sample_categorical row by row. Seeded by drawing a
    single integer from `rng` -- the same external numpy Generator every
    other stochastic decision in this module draws from, so a batched run's
    randomness still traces back to one tracked source rather than
    introducing torch's own untracked global RNG state."""
    generator = torch.Generator(device=probs.device)
    generator.manual_seed(int(rng.integers(0, 2**31 - 1)))
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def simulate_plate_appearances_batch(
    context: GameEngineContext,
    rng: np.random.Generator,
    pitcher_ids: torch.Tensor,
    batter_ids: torch.Tensor,
    game_date,
    season: int,
    park_id: str,
    outs_when_up: torch.Tensor,
    score_diff: torch.Tensor,
    inning: torch.Tensor,
    times_through_order: torch.Tensor,
    bases: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched counterpart to simulate_plate_appearance: resolves one
    terminal plate-appearance outcome for every simulation where
    `active[i]` is True, simultaneously -- each round of the inner loop is
    one batched EventModel call covering every simulation that hasn't yet
    reached a terminal outcome *this plate appearance*, which shrinks every
    round as simulations with shorter counts finish first (a strikeout on
    3 straight pitches drops out of the batch after round 3, while a
    10-pitch at-bat elsewhere keeps its slot occupied). Inactive
    simulations (`active[i]` False -- already finished their whole game)
    are never touched. Returns (outcome_index [len(pitcher_ids)] long --
    an OUTCOME_VOCAB index, meaningless where `active` is False,
    pitches_thrown [len(pitcher_ids)] long)."""
    n = pitcher_ids.shape[0]
    balls = torch.zeros(n, dtype=torch.long)
    strikes = torch.zeros(n, dtype=torch.long)
    pitches_thrown = torch.zeros(n, dtype=torch.long)
    outcome_index = torch.full((n,), -1, dtype=torch.long)
    pa_done = ~active

    ball_idx = OUTCOME_INDEX["ball"]
    foul_idx = OUTCOME_INDEX["foul"]
    called_idx = OUTCOME_INDEX["called_strike"]
    swinging_idx = OUTCOME_INDEX["swinging_strike"]
    unk_idx = OUTCOME_INDEX["UNK"]
    walk_idx = OUTCOME_INDEX["walk"]
    strikeout_idx = OUTCOME_INDEX["strikeout"]
    hipo_idx = OUTCOME_INDEX["hit_into_play_out"]

    for _ in range(MAX_PITCHES_PER_PLATE_APPEARANCE):
        still_going = ~pa_done
        if not bool(still_going.any()):
            break
        idx = still_going.nonzero(as_tuple=True)[0]

        probs = batched_event_outcome_distribution(
            context, pitcher_ids[idx], batter_ids[idx], game_date, season, park_id,
            balls[idx], strikes[idx], outs_when_up[idx], score_diff[idx], inning[idx], times_through_order[idx],
            bases[idx],
        )
        sampled = batched_sample_outcome_indices(probs, rng).cpu()
        pitches_thrown[idx] += 1

        is_ball = sampled == ball_idx
        is_foul = sampled == foul_idx
        is_strike_call = (sampled == called_idx) | (sampled == swinging_idx)
        is_unk = sampled == unk_idx
        is_terminal_direct = ~(is_ball | is_foul | is_strike_call | is_unk)

        sub_balls = balls[idx]
        sub_strikes = strikes[idx]
        sub_balls = torch.where(is_ball, sub_balls + 1, sub_balls)
        sub_strikes = torch.where(is_strike_call, sub_strikes + 1, sub_strikes)
        foul_increment = is_foul & (sub_strikes < 2)  # a foul with 2 strikes doesn't add a 3rd
        sub_strikes = torch.where(foul_increment, sub_strikes + 1, sub_strikes)

        forced_walk = is_ball & (sub_balls >= 4)
        forced_strikeout = is_strike_call & (sub_strikes >= 3)

        this_round_outcome = sampled.clone()
        this_round_outcome = torch.where(forced_walk, torch.full_like(this_round_outcome, walk_idx), this_round_outcome)
        this_round_outcome = torch.where(forced_strikeout, torch.full_like(this_round_outcome, strikeout_idx), this_round_outcome)
        newly_terminal = is_terminal_direct | forced_walk | forced_strikeout

        balls[idx] = sub_balls
        strikes[idx] = sub_strikes
        outcome_index[idx] = torch.where(newly_terminal, this_round_outcome, outcome_index[idx])
        pa_done[idx] = pa_done[idx] | newly_terminal

    still_not_done = active & ~pa_done
    if bool(still_not_done.any()):
        logger.warning(
            "%d simulation(s) exceeded %d pitches in a single plate appearance without a terminal outcome -- "
            "forcing hit_into_play_out.", int(still_not_done.sum()), MAX_PITCHES_PER_PLATE_APPEARANCE,
        )
        outcome_index[still_not_done] = hipo_idx

    return outcome_index, pitches_thrown


def batched_hook_removal_probabilities(
    context: GameEngineContext,
    game_date,
    pitcher_ids: list[int],
    is_starter: list[bool],
    batters_faced: list[int],
    pitch_counts: list[int],
    run_differentials: list[float],
    runner_on_base: list[bool],
    times_through_order: list[int],
) -> np.ndarray:
    """One HookModelPredictor.predict_proba_batch call covering every row
    at once, in place of looping its single-row predict_proba -- replicates
    exactly what predict_proba does per row internally
    (removal_history_features_for's personalized prior, and the pitch-count
    milestone indicators), just building the whole feature DataFrame before
    the one model call rather than a length-1 DataFrame per call. The
    removal-history lookup itself (a dict + np.searchsorted, no model call)
    still loops -- it's cheap and HookModelPredictor exposes no batched
    version of it, but it's not where the per-call cost was coming from."""
    predictor = context.hook_predictor
    cutoff_ns = pd.Timestamp(game_date).value
    n = len(pitcher_ids)
    if n == 0:
        return np.empty(0, dtype="float64")

    avg_batters = np.empty(n)
    avg_pitches = np.empty(n)
    for i in range(n):
        history = predictor.starter_history if is_starter[i] else predictor.reliever_history
        avg_batters[i], avg_pitches[i] = removal_history_features_for(history, pitcher_ids[i], cutoff_ns)

    rows = {
        "batters_faced_so_far": batters_faced,
        "pitch_count": pitch_counts,
        "run_differential": run_differentials,
        "runner_on_base": [float(x) for x in runner_on_base],
        "historical_avg_batters_faced_at_removal": avg_batters,
        "historical_avg_pitch_count_at_removal": avg_pitches,
        "times_through_order": [float(x) for x in times_through_order],
        "is_starter": is_starter,
    }
    for name, threshold in zip(PITCH_COUNT_MILESTONE_FEATURE_NAMES, PITCH_COUNT_MILESTONES):
        rows[name] = [float(pc >= threshold) for pc in pitch_counts]

    return predictor.predict_proba_batch(pd.DataFrame(rows))


def batched_select_replacements(
    context: GameEngineContext,
    game_date,
    removal_requests: list[tuple[int, list[int], set[int]]],
) -> dict[int, int | None]:
    """One BullpenAvailabilityPredictor.predict_proba_batch call covering
    every (game, candidate) row across every game asking for a replacement
    this round, in place of looping its single-row predict_proba once per
    candidate once per game -- `removal_requests` is (game_index, bullpen,
    used_pitcher_ids) per game that needs a replacement selected this
    round; a game with an empty not-yet-used candidate list simply
    contributes no rows and resolves to None, matching
    select_replacement_pitcher's own "no unused bullpen arm" case.

    Never passes a team for the closer-only TEAM_SAVE_OPPORTUNITY feature
    (falls back to 0.0) -- matching select_replacement_pitcher's own
    single-instance call exactly: TeamState has no team-name field
    anywhere in this module, so the single-instance path already never
    passes one either (see BullpenAvailabilityPredictor.predict_proba's
    own team=None fallback). This isn't a shortcut introduced here; it's
    parity with the existing behavior this batched path has to match.
    """
    predictor = context.bullpen_predictor
    history = context.workload_history
    cutoff_ns = pd.Timestamp(game_date).value

    game_indices: list[int] = []
    pitcher_ids: list[int] = []
    for g, bullpen, used in removal_requests:
        for pid in bullpen:
            if pid not in used:
                game_indices.append(g)
                pitcher_ids.append(pid)

    result: dict[int, int | None] = {g: None for g, _, _ in removal_requests}
    if not pitcher_ids:
        return result

    features = np.stack([workload_features_for(history, pid, cutoff_ns) for pid in pitcher_ids])
    rows: dict[str, object] = {name: features[:, i] for i, name in enumerate(WORKLOAD_FEATURE_NAMES)}

    if predictor.closer_model is not None:
        rows["role"] = [
            predictor.roles.get(pid, "unclassified") if predictor.roles is not None else "unclassified"
            for pid in pitcher_ids
        ]
        if predictor.closer_kind == "role_aware_logistic_regression":
            recency = [closer_recency_features_for(history, pid, cutoff_ns) for pid in pitcher_ids]
            rows[CLOSER_RECENCY_FEATURE_NAMES[0]] = [r[0] for r in recency]
            rows[CLOSER_RECENCY_FEATURE_NAMES[1]] = [r[1] for r in recency]
        elif predictor.closer_kind == "closer_only_logistic_regression":
            rows[TEAM_SAVE_OPPORTUNITY_FEATURE_NAME] = [0.0] * len(pitcher_ids)

    probs = predictor.predict_proba_batch(pd.DataFrame(rows))

    best_prob: dict[int, float] = {}
    for g, pid, p in zip(game_indices, pitcher_ids, probs):
        if g not in best_prob or p > best_prob[g]:
            best_prob[g] = p
            result[g] = pid
    return result


def simulate_games_batch(
    count: int,
    home_starter: int,
    away_starter: int,
    home_lineup: list[int],
    away_lineup: list[int],
    home_bullpen: list[int],
    away_bullpen: list[int],
    park_id: str,
    game_date,
    context: GameEngineContext,
    rng: np.random.Generator | None = None,
) -> list[GameResult]:
    """Simulates `count` independent replays of one matchup (identical
    starters/lineups/bullpens/park/date) in parallel, vectorized across a
    batch dimension for the EventModel's per-pitch calls -- see this
    section's module-level comment for what's batched, what isn't, and why.

    Each of the `count` simulations draws its own samples from `rng` at
    every stochastic decision and diverges from the first pitch onward --
    this runs `count` genuinely independent games, not one simulation
    replicated `count` times. Returns a list of `count` GameResult in
    creation order (unrelated to finishing order -- a simulation that
    walks off early is simply held idle, excluded from further updates,
    until the slowest simulation in the batch finishes).
    """
    rng = rng or np.random.default_rng()
    season = pd.Timestamp(game_date).year
    B = count

    home = BatchTeamState.create(B, home_lineup, home_bullpen, home_starter, is_home=True)
    away = BatchTeamState.create(B, away_lineup, away_bullpen, away_starter, is_home=False)
    state = BatchGameState.create(B)
    times_faced: list[dict[tuple[int, int], int]] = [dict() for _ in range(B)]

    while not bool(state.done.all()):
        active = ~state.done
        active_idx = active.nonzero(as_tuple=True)[0].tolist()
        is_top = state.is_top

        batter_ids = torch.where(
            is_top,
            away.lineup.gather(1, (away.batting_index % 9).unsqueeze(1)).squeeze(1),
            home.lineup.gather(1, (home.batting_index % 9).unsqueeze(1)).squeeze(1),
        )
        pitcher_ids = torch.where(is_top, home.current_pitcher_id, away.current_pitcher_id)

        times_through_order = torch.zeros(B, dtype=torch.long)
        for g in active_idx:
            times_through_order[g] = times_faced[g].get((int(pitcher_ids[g]), int(batter_ids[g])), 0)

        batting_score = torch.where(is_top, state.away_score, state.home_score)
        fielding_score = torch.where(is_top, state.home_score, state.away_score)
        score_diff = batting_score - fielding_score

        outcome_index, pitches_thrown = simulate_plate_appearances_batch(
            context, rng, pitcher_ids, batter_ids, game_date, season, park_id,
            state.outs, score_diff, state.inning, times_through_order, state.bases, active,
        )

        home_pitching = is_top & active
        away_pitching = (~is_top) & active
        home.stint_pitch_count = torch.where(home_pitching, home.stint_pitch_count + pitches_thrown, home.stint_pitch_count)
        away.stint_pitch_count = torch.where(away_pitching, away.stint_pitch_count + pitches_thrown, away.stint_pitch_count)
        home.stint_batters_faced = torch.where(home_pitching, home.stint_batters_faced + 1, home.stint_batters_faced)
        away.stint_batters_faced = torch.where(away_pitching, away.stint_batters_faced + 1, away.stint_batters_faced)

        for g in active_idx:
            key = (int(pitcher_ids[g]), int(batter_ids[g]))
            times_faced[g][key] = times_through_order[g].item() + 1

        runs = torch.zeros(B, dtype=torch.long)
        new_outs_delta = torch.zeros(B, dtype=torch.long)
        new_bases_by_game: dict[int, dict[str, int | None]] = {}
        for g in active_idx:
            outcome_str = OUTCOME_VOCAB[int(outcome_index[g])]
            bases_dict = {
                base: (None if int(state.bases[g, i]) == EMPTY_BASE else int(state.bases[g, i]))
                for i, base in enumerate(BASES)
            }
            g_runs, g_new_outs, g_new_bases = apply_outcome(
                context.baserunning_model, rng, bases_dict, outcome_str, int(batter_ids[g]), int(state.outs[g]), season
            )
            runs[g] = g_runs
            new_outs_delta[g] = g_new_outs
            new_bases_by_game[g] = g_new_bases

        state.away_score = torch.where(is_top & active, state.away_score + runs, state.away_score)
        state.home_score = torch.where((~is_top) & active, state.home_score + runs, state.home_score)
        new_outs_total = torch.clamp(state.outs + new_outs_delta, max=3)
        state.outs = torch.where(active, new_outs_total, state.outs)

        for g in active_idx:
            if int(new_outs_total[g]) < 3:
                b = new_bases_by_game[g]
                for i, base in enumerate(BASES):
                    state.bases[g, i] = EMPTY_BASE if b[base] is None else b[base]

        home.batting_index = torch.where((~is_top) & active, home.batting_index + 1, home.batting_index)
        away.batting_index = torch.where(is_top & active, away.batting_index + 1, away.batting_index)

        # Walk-off: the go-ahead run in the bottom of the 9th or later ends
        # that simulation immediately, regardless of the out count on this play.
        walkoff = (~is_top) & active & (state.inning >= 9) & (state.home_score > state.away_score)
        state.done = state.done | walkoff

        ended_half = active & ~walkoff & (state.outs >= 3)
        home_leads_after_top = ended_half & is_top & (state.inning >= 9) & (state.home_score > state.away_score)
        bottom_ended_decided = ended_half & (~is_top) & (state.inning >= 9) & (state.home_score != state.away_score)
        state.done = state.done | home_leads_after_top | bottom_ended_decided

        transitioning = ended_half & ~home_leads_after_top & ~bottom_ended_decided
        state.inning = torch.where(transitioning & (~is_top), state.inning + 1, state.inning)
        new_is_top = torch.where(transitioning, ~is_top, is_top)
        state.outs = torch.where(transitioning, torch.zeros_like(state.outs), state.outs)

        for g in transitioning.nonzero(as_tuple=True)[0].tolist():
            next_team = away if bool(new_is_top[g]) else home
            if int(state.inning[g]) > 9:
                # 2020+ extra-innings rule -- see simulate_game's own comment.
                ghost_runner_id = int(next_team.lineup[g, (int(next_team.batting_index[g]) - 1) % 9])
                state.bases[g, 0] = EMPTY_BASE
                state.bases[g, 1] = ghost_runner_id
                state.bases[g, 2] = EMPTY_BASE
            else:
                state.bases[g] = EMPTY_BASE

        state.is_top = new_is_top

        # Phase 6/5: hook decision, evaluated against the state as it now
        # stands (post-transition) for whichever team just finished
        # pitching this half-inning -- `is_top` here is deliberately the
        # PRE-transition value captured at the top of this loop iteration,
        # not new_is_top: the pitcher whose stint just ended belongs to the
        # half-inning that just happened, even though the score/bases/inning
        # they're evaluated against already reflect the state after it (the
        # same "next at-bat's first pitch" convention hook_model.py trains
        # against -- see maybe_replace_pitcher's own docstring).
        #
        # Batched via batched_hook_removal_probabilities (one
        # HookModelPredictor.predict_proba_batch call covering every active
        # game) and batched_select_replacements (one
        # BullpenAvailabilityPredictor.predict_proba_batch call covering
        # every (game, candidate) row across every game actually removing
        # a pitcher this round) -- replacing what was previously a
        # single-row predict_proba call per active game per round, which
        # profiling showed was ~75% of this function's total wall time at
        # scale (see module docstring's benchmark discussion).
        hook_game_indices: list[int] = []
        hook_pitcher_ids: list[int] = []
        hook_is_starter: list[bool] = []
        hook_batters_faced: list[int] = []
        hook_pitch_counts: list[int] = []
        hook_run_differentials: list[float] = []
        hook_runner_on_base: list[bool] = []
        hook_times_through_order: list[int] = []
        for g in active_idx:
            pitching_is_home = bool(is_top[g])
            pitching_team = home if pitching_is_home else away
            hook_game_indices.append(g)
            hook_pitcher_ids.append(int(pitching_team.current_pitcher_id[g]))
            hook_is_starter.append(int(pitching_team.current_pitcher_id[g]) == int(pitching_team.starter_id[g]))
            hook_batters_faced.append(int(pitching_team.stint_batters_faced[g]))
            hook_pitch_counts.append(int(pitching_team.stint_pitch_count[g]))
            pitcher_team_score = int(state.home_score[g]) if pitching_is_home else int(state.away_score[g])
            opponent_score = int(state.away_score[g]) if pitching_is_home else int(state.home_score[g])
            hook_run_differentials.append(float(pitcher_team_score - opponent_score))
            hook_runner_on_base.append(bool((state.bases[g] != EMPTY_BASE).any()))
            hook_times_through_order.append(int(times_through_order[g]))

        if hook_game_indices:
            removal_probabilities = batched_hook_removal_probabilities(
                context, game_date, hook_pitcher_ids, hook_is_starter, hook_batters_faced, hook_pitch_counts,
                hook_run_differentials, hook_runner_on_base, hook_times_through_order,
            )
            draws = rng.random(len(hook_game_indices))

            removal_requests = []
            for local_i, g in enumerate(hook_game_indices):
                if draws[local_i] >= removal_probabilities[local_i]:
                    continue
                pitching_is_home = bool(is_top[g])
                pitching_team = home if pitching_is_home else away
                removal_requests.append((g, pitching_team.bullpen[g], pitching_team.used_pitcher_ids[g]))

            if removal_requests:
                replacements = batched_select_replacements(context, game_date, removal_requests)
                for g, _, _ in removal_requests:
                    pitching_is_home = bool(is_top[g])
                    pitching_team = home if pitching_is_home else away
                    replacement_id = replacements[g]
                    if replacement_id is None:
                        logger.warning(
                            "Hook model called for removing pitcher_id=%s but no unused bullpen arm remains -- "
                            "keeping them in.", int(pitching_team.current_pitcher_id[g]),
                        )
                        continue
                    pitching_team.current_pitcher_id[g] = replacement_id
                    pitching_team.used_pitcher_ids[g].add(replacement_id)
                    pitching_team.stint_batters_faced[g] = 0
                    pitching_team.stint_pitch_count[g] = 0

    results = []
    for g in range(B):
        winner = "home" if int(state.home_score[g]) > int(state.away_score[g]) else "away"
        results.append(
            GameResult(
                home_score=int(state.home_score[g]), away_score=int(state.away_score[g]),
                innings_played=int(state.inning[g]), winner=winner,
            )
        )
    return results
