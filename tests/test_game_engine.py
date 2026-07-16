import numpy as np
import pandas as pd
import pytest
import torch

from src.data.event_dataset import SITUATIONAL_CONTINUOUS_FEATURES
from src.data.sequence_dataset import OUTCOME_INDEX, OUTCOME_VOCAB
from src.simulation.baserunning import BaserunningConfig, BaserunningModel
from src.simulation.game_engine import (
    GameEngineContext,
    GameState,
    TeamState,
    apply_force_advance,
    apply_outcome,
    build_handedness_lookup,
    maybe_replace_pitcher,
    place_batter,
    resolve_batted_ball,
    sample_categorical,
    select_replacement_pitcher,
    simulate_game,
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

    def predict_proba(self, workload_history, pitcher_id, as_of_date, team=None):
        return self.scores[pitcher_id]


class _FakeHookPredictor:
    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, *args, **kwargs):
        return self.probability


def _fake_context(hook_probability: float, bullpen_scores: dict[int, float]) -> GameEngineContext:
    return GameEngineContext(
        event_model=None,
        park_factor_embedding=None,
        situational_stats={},
        league_rates=pd.DataFrame(),
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


class _FakeParkFactorEmbedding:
    def index_for(self, park_id, season):
        return 0


def _build_scripted_context(outcomes: list[str], baserunning_rates: pd.DataFrame | None = None) -> GameEngineContext:
    if baserunning_rates is None:
        baserunning_rates = pd.DataFrame(columns=["start_base", "outcome", "end_base", "size", "probability"])
    return GameEngineContext(
        event_model=_ScriptedEventModel(outcomes),
        park_factor_embedding=_FakeParkFactorEmbedding(),
        situational_stats={col: (0.0, 1.0) for col in SITUATIONAL_CONTINUOUS_FEATURES},
        league_rates=pd.DataFrame({"season": [2023], "league_hr_rate": [0.1], "league_runs_rate": [4.5]}),
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
