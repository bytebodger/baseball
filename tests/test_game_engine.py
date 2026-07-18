from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.contact_quality import ContactQualityHistory
from src.data.event_dataset import SITUATIONAL_CONTINUOUS_FEATURES
from src.data.park_factors import LeagueRatesIndex
from src.data.sequence_dataset import OUTCOME_INDEX, OUTCOME_VOCAB
from src.models.bullpen_availability import PitcherWorkloadHistory
from src.models.hook_model import PitcherRemovalHistory
from src.simulation.baserunning import BaserunningConfig, BaserunningModel
from src.simulation.game_engine import (
    GameEngineContext,
    GameState,
    TeamState,
    apply_force_advance,
    apply_outcome,
    batched_hook_removal_probabilities,
    batched_select_replacements,
    build_handedness_lookup,
    log_checkpoint_training_metadata,
    maybe_replace_pitcher,
    place_batter,
    resolve_batted_ball,
    sample_categorical,
    select_replacement_pitcher,
    simulate_game,
    simulate_games_batch,
)


# ---------- apply_force_advance (walk / HBP) ----------


def test_apply_force_advance_empty_bases():
    bases = {"1B": None, "2B": None, "3B": None}
    runs, outs, new_bases = apply_force_advance(bases, 99)
    assert (runs, outs) == (0, 0)
    assert new_bases == {"1B": 99, "2B": None, "3B": None}


def test_apply_force_advance_only_forces_runners_behind_the_batter():
    bases = {"1B": None, "2B": 2, "3B": None}
    runs, outs, new_bases = apply_force_advance(bases, 99)
    # 2nd isn't forced (1st was empty) -- unaffected by the walk.
    assert (runs, outs) == (0, 0)
    assert new_bases == {"1B": 99, "2B": 2, "3B": None}


def test_apply_force_advance_cascades_with_bases_loaded():
    bases = {"1B": 1, "2B": 2, "3B": 3}
    runs, outs, new_bases = apply_force_advance(bases, 99)
    assert (runs, outs) == (1, 0)
    assert new_bases == {"1B": 99, "2B": 1, "3B": 2}


# ---------- place_batter ----------


@pytest.mark.parametrize(
    "outcome,expected_base",
    [("single", "1B"), ("double", "2B"), ("triple", "3B")],
)
def test_place_batter_hits(outcome, expected_base):
    new_bases = {"1B": None, "2B": None, "3B": None}
    runs, is_out = place_batter(new_bases, outcome, 42)
    assert (runs, is_out) == (0, 0)
    assert new_bases[expected_base] == 42


def test_place_batter_home_run_scores_without_occupying_a_base():
    new_bases = {"1B": None, "2B": None, "3B": None}
    runs, is_out = place_batter(new_bases, "home_run", 42)
    assert (runs, is_out) == (1, 0)
    assert new_bases == {"1B": None, "2B": None, "3B": None}


def test_place_batter_hit_into_play_out_is_a_batter_out():
    new_bases = {"1B": None, "2B": None, "3B": None}
    runs, is_out = place_batter(new_bases, "hit_into_play_out", 42)
    assert (runs, is_out) == (0, 1)
    assert new_bases == {"1B": None, "2B": None, "3B": None}


# ---------- resolve_batted_ball ----------


class _FakeBaserunningModel:
    def __init__(self, mapping: dict[tuple[str, str], dict[str, float]]):
        self.mapping = mapping

    def advancement_distribution(self, start_base, outcome, runner_id=None, season=None, outs=None):
        return self.mapping.get((start_base, outcome), {})


def test_resolve_batted_ball_scores_and_records_outs():
    fake_model = _FakeBaserunningModel(
        {
            ("3B", "hit_into_play_out"): {"HOME": 1.0},
            ("1B", "hit_into_play_out"): {"OUT": 1.0},
        }
    )
    bases = {"1B": 5, "2B": None, "3B": 6}
    runs, outs, new_bases = resolve_batted_ball(fake_model, np.random.default_rng(0), bases, "hit_into_play_out", outs_when_up=1, season=2023)
    assert runs == 1
    assert outs == 1
    assert new_bases == {"1B": None, "2B": None, "3B": None}


def test_resolve_batted_ball_holds_runner_when_no_empirical_data():
    fake_model = _FakeBaserunningModel({})
    bases = {"1B": 10, "2B": None, "3B": None}
    runs, outs, new_bases = resolve_batted_ball(fake_model, np.random.default_rng(0), bases, "single", outs_when_up=0, season=2023)
    assert (runs, outs) == (0, 0)
    assert new_bases == {"1B": 10, "2B": None, "3B": None}


def test_resolve_batted_ball_bumps_trailing_runner_on_base_collision():
    # Both runners are forced (by this fake model) toward "2B" -- physically
    # impossible for both to land there. The lead runner (processed first,
    # from 3B/2B/1B order) keeps its spot; the trailing runner is bumped
    # forward one additional base.
    fake_model = _FakeBaserunningModel(
        {
            ("2B", "single"): {"2B": 1.0},
            ("1B", "single"): {"2B": 1.0},
        }
    )
    bases = {"1B": 10, "2B": 20, "3B": None}
    runs, outs, new_bases = resolve_batted_ball(fake_model, np.random.default_rng(0), bases, "single", outs_when_up=0, season=2023)
    assert (runs, outs) == (0, 0)
    assert new_bases == {"1B": None, "2B": 20, "3B": 10}


def test_resolve_batted_ball_bump_past_third_scores_a_run():
    fake_model = _FakeBaserunningModel(
        {
            ("3B", "single"): {"3B": 1.0},
            ("2B", "single"): {"3B": 1.0},
        }
    )
    bases = {"1B": None, "2B": 20, "3B": 30}
    runs, outs, new_bases = resolve_batted_ball(fake_model, np.random.default_rng(0), bases, "single", outs_when_up=0, season=2023)
    assert runs == 1  # bumped runner pushed past 3rd scores
    assert outs == 0
    assert new_bases == {"1B": None, "2B": None, "3B": 30}


# ---------- apply_outcome ----------


def test_apply_outcome_strikeout_adds_one_out_and_copies_bases():
    bases = {"1B": 1, "2B": None, "3B": None}
    runs, outs, new_bases = apply_outcome(None, np.random.default_rng(0), bases, "strikeout", batter_id=99, outs_when_up=0, season=2023)
    assert (runs, outs) == (0, 1)
    assert new_bases == bases
    assert new_bases is not bases


def test_apply_outcome_walk_delegates_to_force_advance():
    bases = {"1B": None, "2B": None, "3B": None}
    runs, outs, new_bases = apply_outcome(None, np.random.default_rng(0), bases, "walk", batter_id=99, outs_when_up=0, season=2023)
    assert (runs, outs) == (0, 0)
    assert new_bases == {"1B": 99, "2B": None, "3B": None}


def test_apply_outcome_hit_by_pitch_delegates_to_force_advance():
    bases = {"1B": 1, "2B": None, "3B": None}
    runs, outs, new_bases = apply_outcome(None, np.random.default_rng(0), bases, "hit_by_pitch", batter_id=99, outs_when_up=0, season=2023)
    assert new_bases == {"1B": 99, "2B": 1, "3B": None}


def test_apply_outcome_home_run_scores_batter_plus_existing_runners():
    fake_model = _FakeBaserunningModel({("1B", "home_run"): {"HOME": 1.0}})
    bases = {"1B": 5, "2B": None, "3B": None}
    runs, outs, new_bases = apply_outcome(fake_model, np.random.default_rng(0), bases, "home_run", batter_id=99, outs_when_up=0, season=2023)
    assert runs == 2
    assert outs == 0
    assert new_bases == {"1B": None, "2B": None, "3B": None}


def test_apply_outcome_raises_on_a_non_terminal_outcome():
    bases = {"1B": None, "2B": None, "3B": None}
    with pytest.raises(ValueError):
        apply_outcome(None, np.random.default_rng(0), bases, "ball", batter_id=1, outs_when_up=0, season=2023)


# ---------- sample_categorical ----------


def test_sample_categorical_deterministic_single_outcome():
    rng = np.random.default_rng(0)
    assert sample_categorical({"a": 1.0}, rng) == "a"


def test_sample_categorical_raises_on_non_positive_total():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_categorical({"a": 0.0, "b": 0.0}, rng)


def test_sample_categorical_respects_relative_weights():
    rng = np.random.default_rng(0)
    counts = {"a": 0, "b": 0}
    for _ in range(3000):
        counts[sample_categorical({"a": 0.9, "b": 0.1}, rng)] += 1
    assert counts["a"] > counts["b"]


# ---------- build_handedness_lookup ----------


def test_build_handedness_lookup_picks_most_common():
    pitches = pd.DataFrame(
        [
            {"pitcher_id": 1, "p_throws": "R", "batter_id": 10, "stand": "L"},
            {"pitcher_id": 1, "p_throws": "R", "batter_id": 10, "stand": "L"},
            {"pitcher_id": 1, "p_throws": "L", "batter_id": 10, "stand": "R"},
            {"pitcher_id": 2, "p_throws": "L", "batter_id": 20, "stand": "R"},
        ]
    )
    lookup = build_handedness_lookup(pitches)
    assert lookup["pitcher"][1] == "R"
    assert lookup["pitcher"][2] == "L"
    assert lookup["batter"][10] == "L"
    assert lookup["batter"][20] == "R"


# ---------- TeamState ----------


def test_team_state_rejects_a_non_nine_batter_lineup():
    with pytest.raises(ValueError):
        TeamState(lineup=[1, 2, 3], bullpen=[], starter_id=1, is_home=True)


def test_team_state_initializes_current_pitcher_to_the_starter():
    team = TeamState(lineup=list(range(1, 10)), bullpen=[], starter_id=42, is_home=False)
    assert team.current_pitcher_id == 42
    assert 42 in team.used_pitcher_ids


# ---------- select_replacement_pitcher / maybe_replace_pitcher ----------


class _FakeBullpenPredictor:
    def __init__(self, scores: dict[int, float]):
        self.scores = scores
        self.closer_model = None  # skips the closer-path branching in batched_select_replacements
        self.roles = None
        self.closer_kind = None
        self.team_save_history = None

    def predict_proba(self, workload_history, pitcher_id, as_of_date, team=None):
        return self.scores[pitcher_id]

    def predict_proba_batch(self, examples):
        # Only ever exercised by tests that deliberately set up a real
        # workload_history and override this method themselves (see
        # test_simulate_games_batch_batched_bullpen_selection_picks_the_most_rested_candidate)
        # -- self.scores has no way to identify a row's pitcher_id from the
        # feature-only DataFrame batched_select_replacements builds.
        raise NotImplementedError("this fake's predict_proba_batch is not meaningful without pitcher_id in `examples`")


class _FakeHookPredictor:
    def __init__(self, probability: float):
        self.probability = probability
        empty_history = PitcherRemovalHistory({}, {}, {}, league_avg_batters_faced=3.0, league_avg_pitch_count=50.0)
        self.starter_history = empty_history
        self.reliever_history = empty_history

    def predict_proba(self, *args, **kwargs):
        return self.probability

    def predict_proba_batch(self, examples):
        return np.full(len(examples), self.probability)


def _fake_context(hook_probability: float, bullpen_scores: dict[int, float]) -> GameEngineContext:
    return GameEngineContext(
        event_model=None,
        park_factor_embedding=None,
        situational_stats={},
        league_rates=pd.DataFrame(),
        league_rates_index=None,
        pitcher_contact_quality=None,
        batter_contact_quality=None,
        contact_quality_stats=None,
        pitcher_cache=None,
        batter_cache=None,
        handedness={"pitcher": {}, "batter": {}},
        hook_predictor=_FakeHookPredictor(hook_probability),
        bullpen_predictor=_FakeBullpenPredictor(bullpen_scores),
        workload_history=None,
        baserunning_model=None,
        device=torch.device("cpu"),
    )


def test_select_replacement_pitcher_picks_highest_scored_unused_candidate():
    context = _fake_context(hook_probability=0.0, bullpen_scores={501: 0.2, 502: 0.9, 503: 0.5})
    team = TeamState(lineup=list(range(1, 10)), bullpen=[501, 502, 503], starter_id=99, is_home=True)
    assert select_replacement_pitcher(context, team, "2023-04-01") == 502


def test_select_replacement_pitcher_excludes_already_used_pitchers():
    context = _fake_context(hook_probability=0.0, bullpen_scores={501: 0.2, 502: 0.9})
    team = TeamState(lineup=list(range(1, 10)), bullpen=[501, 502], starter_id=99, is_home=True)
    team.used_pitcher_ids.add(502)
    assert select_replacement_pitcher(context, team, "2023-04-01") == 501


def test_select_replacement_pitcher_returns_none_when_bullpen_exhausted():
    context = _fake_context(hook_probability=0.0, bullpen_scores={})
    team = TeamState(lineup=list(range(1, 10)), bullpen=[], starter_id=99, is_home=True)
    assert select_replacement_pitcher(context, team, "2023-04-01") is None


def test_maybe_replace_pitcher_removes_and_installs_the_top_scored_replacement():
    context = _fake_context(hook_probability=1.0, bullpen_scores={501: 0.2, 502: 0.9, 503: 0.5})
    team = TeamState(lineup=list(range(1, 10)), bullpen=[501, 502, 503], starter_id=99, is_home=True)
    team.stint_batters_faced = 5
    team.stint_pitch_count = 80
    state = GameState(home_score=3, away_score=1)

    maybe_replace_pitcher(context, np.random.default_rng(0), team, state, "2023-04-01", times_through_order=1)

    assert team.current_pitcher_id == 502
    assert team.stint_batters_faced == 0
    assert team.stint_pitch_count == 0
    assert 502 in team.used_pitcher_ids


def test_maybe_replace_pitcher_does_nothing_when_removal_probability_is_zero():
    context = _fake_context(hook_probability=0.0, bullpen_scores={501: 0.9})
    team = TeamState(lineup=list(range(1, 10)), bullpen=[501], starter_id=99, is_home=True)
    team.stint_batters_faced = 5
    team.stint_pitch_count = 80
    state = GameState()

    maybe_replace_pitcher(context, np.random.default_rng(0), team, state, "2023-04-01", times_through_order=1)

    assert team.current_pitcher_id == 99
    assert team.stint_batters_faced == 5
    assert team.stint_pitch_count == 80


def test_maybe_replace_pitcher_keeps_current_pitcher_when_bullpen_exhausted():
    context = _fake_context(hook_probability=1.0, bullpen_scores={})
    team = TeamState(lineup=list(range(1, 10)), bullpen=[], starter_id=99, is_home=True)
    state = GameState()

    maybe_replace_pitcher(context, np.random.default_rng(0), team, state, "2023-04-01", times_through_order=1)

    assert team.current_pitcher_id == 99


# ---------- simulate_game (full end-to-end, fully scripted/deterministic) ----------


class _ScriptedEventModel:
    """Stands in for the real EventModel: ignores the batch entirely and
    returns a logits tensor so sharply peaked at a pre-scripted outcome that
    float32 softmax rounds it to exactly (1.0, 0.0, 0.0, ...) -- sampling is
    then deterministic regardless of rng state, so a whole game's sequence of
    outcomes can be scripted in advance and asserted on exactly."""

    def __init__(self, outcomes: list[str]):
        self._outcomes = list(outcomes)
        self._i = 0

    def __call__(self, batch: dict) -> torch.Tensor:
        assert self._i < len(self._outcomes), "scripted outcome sequence exhausted -- game ran longer than expected"
        outcome = self._outcomes[self._i]
        self._i += 1
        logits = torch.full((1, len(OUTCOME_VOCAB)), -40.0)
        logits[0, OUTCOME_INDEX[outcome]] = 40.0
        return logits


class _FakeEmbeddingCache:
    def get(self, player_id, game_date):
        return torch.zeros(4)

    def get_batch(self, player_ids, game_dates):
        return torch.stack([self.get(p, d) for p, d in zip(player_ids, game_dates)])


class _FakeParkFactorEmbedding:
    def index_for(self, park_id, season):
        return 0


def _empty_contact_quality() -> ContactQualityHistory:
    return ContactQualityHistory(
        {}, {}, {}, league_avg_exit_velo=90.0, league_avg_hard_hit_rate=0.3,
        babip_dates_by_player={}, babip_hit_by_player={}, league_avg_babip=0.3,
    )


def _dummy_contact_quality_stats() -> dict[str, tuple[float, float]]:
    return {"pitcher_exit_velo": (90.0, 1.0), "batter_exit_velo": (90.0, 1.0)}


def _build_scripted_context(outcomes: list[str], baserunning_rates: pd.DataFrame | None = None) -> GameEngineContext:
    if baserunning_rates is None:
        baserunning_rates = pd.DataFrame(columns=["start_base", "outcome", "end_base", "size", "probability"])
    league_rates = pd.DataFrame({"season": [2023], "league_hr_rate": [0.1], "league_runs_rate": [4.5]})
    return GameEngineContext(
        event_model=_ScriptedEventModel(outcomes),
        park_factor_embedding=_FakeParkFactorEmbedding(),
        situational_stats={col: (0.0, 1.0) for col in SITUATIONAL_CONTINUOUS_FEATURES},
        league_rates=league_rates,
        league_rates_index=LeagueRatesIndex(league_rates),
        pitcher_contact_quality=_empty_contact_quality(),
        batter_contact_quality=_empty_contact_quality(),
        contact_quality_stats=_dummy_contact_quality_stats(),
        pitcher_cache=_FakeEmbeddingCache(),
        batter_cache=_FakeEmbeddingCache(),
        handedness={"pitcher": {}, "batter": {}},
        hook_predictor=_FakeHookPredictor(probability=0.0),
        bullpen_predictor=_FakeBullpenPredictor(scores={}),
        workload_history=None,
        baserunning_model=BaserunningModel(baserunning_rates, BaserunningConfig()),
        device=torch.device("cpu"),
    )


def test_simulate_game_home_wins_without_a_bottom_ninth():
    # Away strikes out every time (never scores). Home hits a home run in the
    # bottom of the 1st (their only run) and strikes out otherwise -- a home
    # run doesn't record an out, so bottom 1 needs 3 *more* strikeouts after
    # it to actually reach 3 outs. By the top of the 9th home already leads
    # 1-0, so the bottom 9th is never played -- regulation ends right there.
    outcomes = []
    outcomes += ["strikeout"] * 3  # top 1
    outcomes += ["home_run", "strikeout", "strikeout", "strikeout"]  # bottom 1
    for _ in range(7):  # innings 2-8, both halves
        outcomes += ["strikeout"] * 3
        outcomes += ["strikeout"] * 3
    outcomes += ["strikeout"] * 3  # top 9

    context = _build_scripted_context(outcomes)
    result = simulate_game(
        home_starter=1,
        away_starter=2,
        home_lineup=list(range(101, 110)),
        away_lineup=list(range(201, 210)),
        home_bullpen=[301, 302],
        away_bullpen=[401, 402],
        park_id="COL",
        game_date="2023-04-01",
        context=context,
        rng=np.random.default_rng(0),
    )

    assert result.home_score == 1
    assert result.away_score == 0
    assert result.innings_played == 9
    assert result.winner == "home"


def test_simulate_game_extra_innings_ghost_runner_walk_off():
    # 9 full, scoreless innings (everyone strikes out) -> tied, extra
    # innings. Top 10: away strikes out (their own ghost runner stranded).
    # Bottom 10: home's ghost runner (on 2nd) scores on the very first
    # batter's single, ending the game immediately as a walk-off.
    outcomes = []
    for _ in range(9):
        outcomes += ["strikeout"] * 3
        outcomes += ["strikeout"] * 3
    outcomes += ["strikeout"] * 3  # top 10
    outcomes += ["single"]  # bottom 10, first batter -- scores the ghost runner

    baserunning_rates = pd.DataFrame(
        [{"start_base": "2B", "outcome": "single", "end_base": "HOME", "size": 1, "probability": 1.0}]
    )
    context = _build_scripted_context(outcomes, baserunning_rates)
    result = simulate_game(
        home_starter=1,
        away_starter=2,
        home_lineup=list(range(101, 110)),
        away_lineup=list(range(201, 210)),
        home_bullpen=[301, 302],
        away_bullpen=[401, 402],
        park_id="COL",
        game_date="2023-04-01",
        context=context,
        rng=np.random.default_rng(0),
    )

    assert result.home_score == 1
    assert result.away_score == 0
    assert result.innings_played == 10
    assert result.winner == "home"


# ---------- simulate_game: verbose play-by-play logging ----------


def _short_game_outcomes() -> list[str]:
    outcomes = []
    outcomes += ["strikeout"] * 3  # top 1
    outcomes += ["home_run", "strikeout", "strikeout", "strikeout"]  # bottom 1
    for _ in range(7):
        outcomes += ["strikeout"] * 3
        outcomes += ["strikeout"] * 3
    outcomes += ["strikeout"] * 3  # top 9
    return outcomes


def _run_short_game(context, verbose: bool):
    return simulate_game(
        home_starter=1,
        away_starter=2,
        home_lineup=list(range(101, 110)),
        away_lineup=list(range(201, 210)),
        home_bullpen=[301, 302],
        away_bullpen=[401, 402],
        park_id="COL",
        game_date="2023-04-01",
        context=context,
        rng=np.random.default_rng(0),
        verbose=verbose,
    )


def test_simulate_game_verbose_false_logs_nothing_by_default(caplog):
    context = _build_scripted_context(_short_game_outcomes())
    with caplog.at_level("INFO", logger="src.simulation.game_engine"):
        _run_short_game(context, verbose=False)
    assert caplog.messages == []


def test_simulate_game_verbose_true_logs_every_plate_appearance(caplog):
    context = _build_scripted_context(_short_game_outcomes())
    with caplog.at_level("INFO", logger="src.simulation.game_engine"):
        result = _run_short_game(context, verbose=True)

    play_lines = [m for m in caplog.messages if "pitcher=" in m and "batter=" in m]
    # One logged line per plate appearance: 3 (top1) + 4 (bottom1) + 7*6 (innings 2-8) + 3 (top9) = 52.
    assert len(play_lines) == 52
    assert any("home_run" in line for line in play_lines)
    assert any("strikeout" in line for line in play_lines)


def test_simulate_game_verbose_true_logs_game_start_and_final_result(caplog):
    context = _build_scripted_context(_short_game_outcomes())
    with caplog.at_level("INFO", logger="src.simulation.game_engine"):
        result = _run_short_game(context, verbose=True)

    assert any("Game start" in m for m in caplog.messages)
    assert any("no bottom half needed" in m and str(result.away_score) in m and str(result.home_score) in m for m in caplog.messages)


def test_simulate_game_verbose_true_logs_half_inning_transitions(caplog):
    context = _build_scripted_context(_short_game_outcomes())
    with caplog.at_level("INFO", logger="src.simulation.game_engine"):
        _run_short_game(context, verbose=True)

    transition_lines = [m for m in caplog.messages if m.startswith("--- ")]
    # "--- Top 1 ---" (game start) plus one per subsequent half-inning
    # actually played: bottom 1 through top 9 = 16 more transitions.
    assert len(transition_lines) == 17
    assert transition_lines[0] == "--- Top 1 ---"


def test_simulate_game_verbose_true_logs_ghost_runner_placement(caplog):
    outcomes = []
    for _ in range(9):
        outcomes += ["strikeout"] * 3
        outcomes += ["strikeout"] * 3
    outcomes += ["strikeout"] * 3  # top 10
    outcomes += ["single"]  # bottom 10 walk-off

    baserunning_rates = pd.DataFrame(
        [{"start_base": "2B", "outcome": "single", "end_base": "HOME", "size": 1, "probability": 1.0}]
    )
    context = _build_scripted_context(outcomes, baserunning_rates)
    with caplog.at_level("INFO", logger="src.simulation.game_engine"):
        result = simulate_game(
            home_starter=1, away_starter=2,
            home_lineup=list(range(101, 110)), away_lineup=list(range(201, 210)),
            home_bullpen=[301, 302], away_bullpen=[401, 402],
            park_id="COL", game_date="2023-04-01",
            context=context, rng=np.random.default_rng(0), verbose=True,
        )

    assert any("ghost runner" in m for m in caplog.messages)
    assert any("Walk-off" in m for m in caplog.messages)
    assert result.winner == "home"


# ---------- simulate_games_batch ----------


class _ScriptedEventModelBatch:
    """Batch-size-agnostic counterpart to _ScriptedEventModel: broadcasts
    the same scripted, sharply-peaked-one-hot outcome to every row of
    whatever batch size it's called with. Since every row gets an
    identical deterministic distribution, every simulation in the batch
    follows the exact same trajectory regardless of rng -- which is what
    makes it possible to check a batched run's result against a known
    single-game trajectory scripted with the same outcome list."""

    def __init__(self, outcomes: list[str]):
        self._outcomes = list(outcomes)
        self._i = 0

    def __call__(self, batch: dict) -> torch.Tensor:
        n = batch["pitcher_embedding"].shape[0]
        assert self._i < len(self._outcomes), "scripted outcome sequence exhausted"
        outcome = self._outcomes[self._i]
        self._i += 1
        logits = torch.full((n, len(OUTCOME_VOCAB)), -40.0)
        logits[:, OUTCOME_INDEX[outcome]] = 40.0
        return logits


class _MixedEventModel:
    """Every pitch is an even coin flip between two terminal outcomes
    (strikeout / walk) -- used to verify a batch of simulations genuinely
    diverges (real per-row randomness from torch.multinomial), rather than
    all following one shared path. strikeout/walk deliberately avoid any
    batted-ball outcome: those get resolved through the baserunning model,
    and this test's baserunning_rates table is empty by design (see
    _build_scripted_batch_context), so a batted ball would fall back to
    "hold the runner in place" -- a fallback real Phase 7 data would never
    actually need for a start_base a runner is forced off of (e.g. 1B on a
    single), since that combination never appears in real advancement
    rates. Walks sidestep the baserunning model entirely (deterministic
    force logic), so this mix can't trigger that synthetic-fixture edge
    case."""

    def __call__(self, batch: dict) -> torch.Tensor:
        n = batch["pitcher_embedding"].shape[0]
        logits = torch.full((n, len(OUTCOME_VOCAB)), -40.0)
        logits[:, OUTCOME_INDEX["strikeout"]] = 0.0
        logits[:, OUTCOME_INDEX["walk"]] = 0.0
        return logits


def _build_scripted_batch_context(event_model, baserunning_rates: pd.DataFrame | None = None) -> GameEngineContext:
    if baserunning_rates is None:
        baserunning_rates = pd.DataFrame(columns=["start_base", "outcome", "end_base", "size", "probability"])
    league_rates = pd.DataFrame({"season": [2023], "league_hr_rate": [0.1], "league_runs_rate": [4.5]})
    return GameEngineContext(
        event_model=event_model,
        park_factor_embedding=_FakeParkFactorEmbedding(),
        situational_stats={col: (0.0, 1.0) for col in SITUATIONAL_CONTINUOUS_FEATURES},
        league_rates=league_rates,
        league_rates_index=LeagueRatesIndex(league_rates),
        pitcher_contact_quality=_empty_contact_quality(),
        batter_contact_quality=_empty_contact_quality(),
        contact_quality_stats=_dummy_contact_quality_stats(),
        pitcher_cache=_FakeEmbeddingCache(),
        batter_cache=_FakeEmbeddingCache(),
        handedness={"pitcher": {}, "batter": {}},
        hook_predictor=_FakeHookPredictor(probability=0.0),
        bullpen_predictor=_FakeBullpenPredictor(scores={}),
        workload_history=None,
        baserunning_model=BaserunningModel(baserunning_rates, BaserunningConfig()),
        device=torch.device("cpu"),
    )


def _batch_game_kwargs(count: int, context: GameEngineContext, rng=None) -> dict:
    return dict(
        count=count,
        home_starter=1, away_starter=2,
        home_lineup=list(range(101, 110)), away_lineup=list(range(201, 210)),
        home_bullpen=[301, 302], away_bullpen=[401, 402],
        park_id="COL", game_date="2023-04-01",
        context=context, rng=rng or np.random.default_rng(0),
    )


def test_simulate_games_batch_returns_the_requested_count():
    context = _build_scripted_batch_context(_ScriptedEventModelBatch(_short_game_outcomes()))
    results = simulate_games_batch(**_batch_game_kwargs(5, context))
    assert len(results) == 5


def test_simulate_games_batch_matches_single_game_deterministic_trajectory():
    # A fully shared one-hot script makes every simulation in the batch
    # follow the identical path -- all 5 results must equal each other and
    # the already-verified single-game result for the same script
    # (test_simulate_game_home_wins_without_a_bottom_ninth: home 1, away 0,
    # 9 innings, home wins).
    context = _build_scripted_batch_context(_ScriptedEventModelBatch(_short_game_outcomes()))
    results = simulate_games_batch(**_batch_game_kwargs(5, context))

    for result in results:
        assert result.home_score == 1
        assert result.away_score == 0
        assert result.innings_played == 9
        assert result.winner == "home"


def test_simulate_games_batch_extra_innings_matches_single_game():
    outcomes = []
    for _ in range(9):
        outcomes += ["strikeout"] * 3
        outcomes += ["strikeout"] * 3
    outcomes += ["strikeout"] * 3  # top 10
    outcomes += ["single"]  # bottom 10 walk-off

    baserunning_rates = pd.DataFrame(
        [{"start_base": "2B", "outcome": "single", "end_base": "HOME", "size": 1, "probability": 1.0}]
    )
    context = _build_scripted_batch_context(_ScriptedEventModelBatch(outcomes), baserunning_rates)
    results = simulate_games_batch(**_batch_game_kwargs(4, context))

    for result in results:
        assert result.home_score == 1
        assert result.away_score == 0
        assert result.innings_played == 10
        assert result.winner == "home"


def test_simulate_games_batch_produces_independent_diverging_results():
    context = _build_scripted_batch_context(_MixedEventModel())
    results = simulate_games_batch(**_batch_game_kwargs(24, context, rng=np.random.default_rng(7)))

    assert len(results) == 24
    # Real per-simulation randomness: not every game should land on the
    # exact same final score with a 50/50 strikeout/walk mix.
    distinct_scorelines = {(r.home_score, r.away_score) for r in results}
    assert len(distinct_scorelines) > 1
    for r in results:
        assert r.innings_played >= 9
        assert r.home_score >= 0 and r.away_score >= 0
        assert r.winner in ("home", "away")


def test_simulate_games_batch_matches_simulate_game_run_by_run():
    # The same deterministic script fed to simulate_game one game at a time
    # must produce identical per-game results to feeding it through
    # simulate_games_batch -- same underlying decision logic (apply_outcome,
    # maybe_replace_pitcher), just vectorized differently.
    single_context = _build_scripted_context(_short_game_outcomes())
    single_result = simulate_game(
        home_starter=1, away_starter=2,
        home_lineup=list(range(101, 110)), away_lineup=list(range(201, 210)),
        home_bullpen=[301, 302], away_bullpen=[401, 402],
        park_id="COL", game_date="2023-04-01",
        context=single_context, rng=np.random.default_rng(0),
    )

    batch_context = _build_scripted_batch_context(_ScriptedEventModelBatch(_short_game_outcomes()))
    batch_results = simulate_games_batch(**_batch_game_kwargs(3, batch_context))

    for batch_result in batch_results:
        assert batch_result == single_result


def test_simulate_games_batch_attributes_hook_decisions_to_the_correct_team(caplog):
    # Regression test: an earlier version of the batched hook-decision path
    # had pitching_is_home inverted, misattributing every removal decision
    # to the wrong team -- silently, since every other batch test uses
    # hook_probability=0.0, which never actually exercises team identity
    # (a decision that's never removed never reveals which team it thought
    # was pitching). Top of inning 1: home should be pitching
    # (simulate_game's own convention -- batting_team, pitching_team =
    # (away, home) if state.is_top else (home, away)). An empty
    # home_bullpen with hook_probability=1.0 must produce "no unused
    # bullpen arm" warnings naming home_starter (pitcher_id=1), never
    # away_starter (pitcher_id=2).
    context = _build_scripted_batch_context(_ScriptedEventModelBatch(_short_game_outcomes()))
    context.hook_predictor = _FakeHookPredictor(probability=1.0)

    kwargs = _batch_game_kwargs(1, context)
    # Both bullpens deliberately empty: every removal attempt (for either
    # team, whenever its turn to pitch comes up) hits
    # batched_select_replacements' "no candidates at all" fast path, which
    # returns None without needing a real workload_history -- keeping this
    # test focused purely on team attribution, not bullpen-scoring plumbing
    # (see the batched_select_replacements-specific tests for that).
    kwargs["home_bullpen"] = []
    kwargs["away_bullpen"] = []

    with caplog.at_level("WARNING", logger="src.simulation.game_engine"):
        result = simulate_games_batch(**kwargs)

    warnings = [m for m in caplog.messages if "no unused bullpen arm remains" in m]
    assert warnings, "expected 'no unused bullpen arm' warnings (both bullpens are empty)"
    assert any("pitcher_id=1" in m for m in warnings), f"expected some warnings for home_starter (pitcher_id=1): {warnings}"
    assert any("pitcher_id=2" in m for m in warnings), f"expected some warnings for away_starter (pitcher_id=2): {warnings}"
    assert result[0].home_score == 1 and result[0].away_score == 0  # unaffected: same script as the deterministic test


# ---------- batched_hook_removal_probabilities / batched_select_replacements (direct unit tests) ----------


def test_batched_hook_removal_probabilities_returns_one_probability_per_row():
    context = _build_scripted_batch_context(_ScriptedEventModelBatch([]))
    context.hook_predictor = _FakeHookPredictor(probability=0.37)

    probs = batched_hook_removal_probabilities(
        context, "2023-04-01",
        pitcher_ids=[1, 2, 3], is_starter=[True, False, True],
        batters_faced=[5, 10, 1], pitch_counts=[80, 20, 5],
        run_differentials=[1.0, -2.0, 0.0], runner_on_base=[True, False, True],
        times_through_order=[1, 0, 0],
    )
    assert list(probs) == pytest.approx([0.37, 0.37, 0.37])


def test_batched_hook_removal_probabilities_empty_input_returns_empty_array():
    context = _build_scripted_batch_context(_ScriptedEventModelBatch([]))
    context.hook_predictor = _FakeHookPredictor(probability=0.5)
    probs = batched_hook_removal_probabilities(context, "2023-04-01", [], [], [], [], [], [], [])
    assert len(probs) == 0


class _MostRestedBullpenPredictor:
    closer_model = None

    def predict_proba_batch(self, examples):
        return examples["days_since_last_appearance"].to_numpy(dtype="float64")


def _rested_workload_history(rest_days_by_pitcher: dict[int, int], as_of: str = "2023-04-01") -> PitcherWorkloadHistory:
    as_of_ns = pd.Timestamp(as_of).value
    day_ns = 86_400_000_000_000
    return PitcherWorkloadHistory(
        dates_by_pitcher={pid: np.array([as_of_ns - days * day_ns], dtype="int64") for pid, days in rest_days_by_pitcher.items()},
        pitches_by_pitcher={pid: np.array([20.0]) for pid in rest_days_by_pitcher},
        save_dates_by_pitcher={pid: np.array([], dtype="int64") for pid in rest_days_by_pitcher},
        light_dates_by_pitcher={pid: np.array([], dtype="int64") for pid in rest_days_by_pitcher},
    )


def test_batched_select_replacements_picks_the_most_rested_candidate_per_game():
    context = _build_scripted_batch_context(_ScriptedEventModelBatch([]))
    context.bullpen_predictor = _MostRestedBullpenPredictor()
    context.workload_history = _rested_workload_history({401: 12, 402: 2, 501: 5, 502: 20})

    removal_requests = [
        (0, [401, 402], set()),  # game 0: 401 is more rested (12 days) than 402 (2 days)
        (1, [501, 502], set()),  # game 1: 502 is more rested (20 days) than 501 (5 days)
    ]
    replacements = batched_select_replacements(context, "2023-04-01", removal_requests)
    assert replacements == {0: 401, 1: 502}


def test_batched_select_replacements_excludes_already_used_candidates():
    context = _build_scripted_batch_context(_ScriptedEventModelBatch([]))
    context.bullpen_predictor = _MostRestedBullpenPredictor()
    context.workload_history = _rested_workload_history({401: 12, 402: 2})

    removal_requests = [(0, [401, 402], {401})]  # 401 (the more-rested one) already used
    replacements = batched_select_replacements(context, "2023-04-01", removal_requests)
    assert replacements == {0: 402}


def test_batched_select_replacements_returns_none_when_no_candidates_remain():
    context = _build_scripted_batch_context(_ScriptedEventModelBatch([]))
    context.bullpen_predictor = _MostRestedBullpenPredictor()
    context.workload_history = _rested_workload_history({401: 12})

    removal_requests = [(0, [401], {401}), (1, [], set())]
    replacements = batched_select_replacements(context, "2023-04-01", removal_requests)
    assert replacements == {0: None, 1: None}


# ---------- log_checkpoint_training_metadata ----------


def test_log_checkpoint_training_metadata_logs_metadata_when_present(caplog):
    ckpt = {"epoch": 9, "val_loss": 1.605, "training_metadata": {"aux_loss_weight": 0.1, "seed": None}}
    with caplog.at_level("INFO", logger="src.simulation.game_engine"):
        log_checkpoint_training_metadata(Path("checkpoints/event_model_full_best.pt"), ckpt)
    assert any("Checkpoint training metadata" in m and "aux_loss_weight" in m for m in caplog.messages)
    assert not any("no training_metadata" in m for m in caplog.messages)


def test_log_checkpoint_training_metadata_warns_when_absent(caplog):
    ckpt = {"epoch": 3, "val_loss": 1.62}
    with caplog.at_level("WARNING", logger="src.simulation.game_engine"):
        log_checkpoint_training_metadata(Path("checkpoints/event_model_full_best.pt"), ckpt)
    assert any("no training_metadata" in m for m in caplog.messages)
