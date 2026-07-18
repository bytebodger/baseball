import numpy as np
import pandas as pd
import pytest

from src.simulation.baserunning import (
    BASE_ADVANCE_ORDER,
    BATTED_BALL_OUTCOMES,
    BaserunningConfig,
    BaserunningModel,
    SprintSpeedHistory,
    adjust_for_sprint_speed,
    build_runner_transitions,
    build_sprint_speed_history,
    compare_advancement_rates_by_outs,
    compute_league_advancement_rates,
    compute_league_advancement_rates_by_outs,
    league_avg_sprint_speed_for,
    load_model,
    main as baserunning_main,
    save_model,
    sprint_speed_for,
)


def _row(game_pk, at_bat_number, home_score, away_score, on_1b, on_2b, on_3b, outcome,
         inning_topbot="Bot", game_date="2023-04-01", season=2023, home_team="DET", away_team="CLE",
         outs_when_up=0):
    return {
        "game_pk": game_pk, "at_bat_number": at_bat_number, "pitch_number": 1,
        "game_date": pd.Timestamp(game_date), "season": season, "inning_topbot": inning_topbot,
        "home_team": home_team, "away_team": away_team,
        "home_score": home_score, "away_score": away_score,
        "on_1b": on_1b, "on_2b": on_2b, "on_3b": on_3b, "outcome": outcome,
        "outs_when_up": outs_when_up,
    }


def _transitions_fixture() -> pd.DataFrame:
    """Every row is "Bot" (home team DET batting), so home_score alone
    tracks the batting team's runs across a play. Each game_pk is an
    independent 2-at-bat scenario: the first row is the transition under
    test, the second supplies the real post-play state (base occupancy +
    score) the first row's runner(s) are resolved against.
    """
    rows = [
        # 1: runner on 2nd (555) scores on a single.
        _row(1, 1, 0, 0, None, 555, None, "single"),
        _row(1, 2, 1, 0, 700, None, None, "strikeout"),
        # 2: runner on 1st (700-a) held at 1st on a ball in play (out).
        _row(2, 1, 0, 0, 700, None, None, "hit_into_play_out"),
        _row(2, 2, 0, 0, 700, None, None, "strikeout"),
        # 3: runner on 1st (800) advances to 3rd on a single (aggressive).
        _row(3, 1, 0, 0, 800, None, None, "single"),
        _row(3, 2, 0, 0, 801, None, 800, "strikeout"),
        # 4: runner on 1st (900) advances to only 2nd on a single (conservative).
        _row(4, 1, 0, 0, 900, None, None, "single"),
        _row(4, 2, 0, 0, 901, 900, None, "strikeout"),
        # 5: runner on 3rd (111) scores on a ball in play (sac-fly-like).
        _row(5, 1, 3, 0, None, None, 111, "hit_into_play_out"),
        _row(5, 2, 4, 0, None, None, None, "strikeout"),
        # 6: runner on 3rd (222) held at 3rd on a ball in play.
        _row(6, 1, 0, 0, None, None, 222, "hit_into_play_out"),
        _row(6, 2, 0, 0, None, None, 222, "strikeout"),
        # 7: runner on 3rd (333) put out on a ball in play (score unchanged).
        _row(7, 1, 2, 0, None, None, 333, "hit_into_play_out"),
        _row(7, 2, 2, 0, None, None, None, "strikeout"),
        # 8: no next at-bat at all -- must be excluded regardless of outcome/runners.
        _row(8, 1, 0, 0, None, 999, None, "single"),
        # 9: strikeout with a runner on base -- not a batted-ball outcome, excluded.
        _row(9, 1, 0, 0, None, None, 888, "strikeout"),
        _row(9, 2, 0, 0, None, None, 888, "strikeout"),
        # 10: bases loaded, ball in play, 2 of 3 vanished runners score -- ambiguous, all 3 dropped.
        # Player IDs deliberately distinct from every other scenario's, so a
        # collision can't accidentally make an ambiguous-dropped runner
        # reappear "legitimately" via an unrelated game.
        _row(10, 1, 5, 0, 2001, 2002, 2003, "hit_into_play_out"),
        _row(10, 2, 7, 0, None, None, None, "strikeout"),
        # 11: another 1st->2nd single, to give compute_league_advancement_rates a real 2:1 mix.
        _row(11, 1, 0, 0, 950, None, None, "single"),
        _row(11, 2, 0, 0, 951, 950, None, "strikeout"),
        # 12: runner on 3rd (444) vanishes with the score unchanged, but the
        # half-inning ends on this very play (inning_topbot flips Bot->Top on
        # the next at-bat). This is a stranded runner, not a basepath out --
        # must be dropped, not labeled OUT.
        _row(12, 1, 2, 0, None, None, 444, "hit_into_play_out", inning_topbot="Bot"),
        _row(12, 2, 2, 0, None, None, None, "strikeout", inning_topbot="Top"),
    ]
    return pd.DataFrame(rows)


# ---------- build_runner_transitions ----------


def test_runner_on_2nd_scores_on_single():
    transitions = build_runner_transitions(_transitions_fixture())
    row = transitions[(transitions["start_base"] == "2B") & (transitions["outcome"] == "single")]
    assert row["end_base"].tolist() == ["HOME"]


def test_runner_on_1st_held_on_ball_in_play():
    transitions = build_runner_transitions(_transitions_fixture())
    subset = transitions[(transitions["start_base"] == "1B") & (transitions["outcome"] == "hit_into_play_out")]
    assert subset["end_base"].tolist() == ["1B"]


def test_runner_on_1st_advances_to_3rd_or_2nd_on_single():
    transitions = build_runner_transitions(_transitions_fixture())
    subset = transitions[(transitions["start_base"] == "1B") & (transitions["outcome"] == "single")]
    assert sorted(subset["end_base"].tolist()) == ["2B", "2B", "3B"]


def test_runner_on_3rd_scores_on_ball_in_play():
    transitions = build_runner_transitions(_transitions_fixture())
    subset = transitions[(transitions["start_base"] == "3B") & (transitions["outcome"] == "hit_into_play_out")]
    assert sorted(subset["end_base"].tolist()) == ["3B", "HOME", "OUT"]


def test_no_next_at_bat_is_excluded():
    transitions = build_runner_transitions(_transitions_fixture())
    # game_pk 8's runner (999) has no following at-bat to resolve their fate
    # against, so they must never appear at all -- not even as an ambiguous
    # drop, since a drop still requires an outcome to have been considered.
    assert 999 not in transitions["runner_id"].to_numpy()


def test_non_batted_ball_outcomes_generate_no_transitions():
    transitions = build_runner_transitions(_transitions_fixture())
    assert "strikeout" not in transitions["outcome"].unique()


def test_ambiguous_mixed_fate_play_drops_all_involved_runners():
    transitions = build_runner_transitions(_transitions_fixture())
    # Bases-loaded game_pk=10 scored 2 of 3 vanished runners -- an
    # unresolvable mix, so none of 444/555/666 should appear at all.
    assert not {2001, 2002, 2003} & set(transitions["runner_id"].tolist())
    # Total count matches exactly the unambiguous scenarios: 1 (game 1) + 1
    # (game 2) + 1 (game 3) + 1 (game 4) + 1 (game 5) + 1 (game 6) + 1
    # (game 7) + 0 (game 8, no next at-bat) + 0 (game 9, not a batted ball)
    # + 0 (game 10, ambiguous) + 1 (game 11) + 0 (game 12, stranded) = 8.
    assert len(transitions) == 8


def test_stranded_runner_at_half_inning_end_is_dropped_not_out():
    transitions = build_runner_transitions(_transitions_fixture())
    # game_pk=12's runner (444) vanishes with score unchanged, but the half
    # inning ends on the same play (inning_topbot flips on the next at-bat)
    # -- they were left stranded, not thrown out on the bases, so they must
    # not appear in the transitions table at all (not even mislabeled OUT).
    assert 444 not in transitions["runner_id"].to_numpy()
    # Contrast with game_pk=7, where the half inning continues (inning_topbot
    # stays "Bot") and the score is also unchanged -- that IS a genuine
    # basepath out and must still be classified as such.
    row = transitions[transitions["runner_id"] == 333]
    assert row["end_base"].tolist() == ["OUT"]


# ---------- compute_league_advancement_rates ----------


def test_compute_league_advancement_rates_probabilities_sum_to_one_per_group():
    transitions = build_runner_transitions(_transitions_fixture())
    rates = compute_league_advancement_rates(transitions)
    totals = rates.groupby(["start_base", "outcome"])["probability"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)


def test_compute_league_advancement_rates_matches_hand_computed_ratio():
    transitions = build_runner_transitions(_transitions_fixture())
    rates = compute_league_advancement_rates(transitions)
    subset = rates[(rates["start_base"] == "1B") & (rates["outcome"] == "single")].set_index("end_base")
    assert subset.loc["2B", "probability"] == pytest.approx(2 / 3)
    assert subset.loc["3B", "probability"] == pytest.approx(1 / 3)


def test_compute_league_advancement_rates_third_to_home_distribution():
    transitions = build_runner_transitions(_transitions_fixture())
    rates = compute_league_advancement_rates(transitions)
    subset = rates[(rates["start_base"] == "3B") & (rates["outcome"] == "hit_into_play_out")].set_index("end_base")
    assert subset.loc["HOME", "probability"] == pytest.approx(1 / 3)
    assert subset.loc["3B", "probability"] == pytest.approx(1 / 3)
    assert subset.loc["OUT", "probability"] == pytest.approx(1 / 3)


# ---------- outs-conditioned rates ----------


def _outs_fixture() -> pd.DataFrame:
    """Three independent 3rd-base/hit_into_play_out scenarios, one per out
    count, isolating how the same nominal situation ("runner on 3rd
    vanishes, score unchanged") resolves differently depending on outs:
    at outs 0 and 1 the half-inning continues, so it's a genuine OUT; at
    outs 2 this at-bat's own out is unavoidably the batting team's 3rd, so
    the half-inning ends and the runner is stranded (dropped), not OUT --
    the selection effect compute_league_advancement_rates_by_outs' and the
    module docstring's both describe.
    """
    rows = [
        # outs=0 -> outs=1 after the play: half-inning continues -> genuine OUT.
        _row(101, 1, 0, 0, None, None, 111, "hit_into_play_out", outs_when_up=0),
        _row(101, 2, 0, 0, None, None, None, "strikeout", outs_when_up=1),
        # outs=1 -> outs=2 after the play: half-inning continues -> genuine OUT.
        _row(102, 1, 0, 0, None, None, 222, "hit_into_play_out", outs_when_up=1),
        _row(102, 2, 0, 0, None, None, None, "strikeout", outs_when_up=2),
        # outs=2: this out is unavoidably the 3rd -- half-inning ends
        # (inning_topbot flips) -> stranded, dropped, NOT counted as OUT.
        _row(103, 1, 0, 0, None, None, 333, "hit_into_play_out", outs_when_up=2, inning_topbot="Bot"),
        _row(103, 2, 0, 0, None, None, None, "strikeout", outs_when_up=0, inning_topbot="Top"),
        # outs=2, but this runner scores (a sac-fly-like play) -- gives the
        # outs=2 group a real, non-dropped observation (a HOME transition),
        # so the pivot table has a genuine outs_2 column to compare against,
        # distinct from "outs=2 never appears in the data at all."
        _row(104, 1, 0, 0, None, None, 444, "hit_into_play_out", outs_when_up=2, inning_topbot="Bot"),
        _row(104, 2, 1, 0, None, None, None, "strikeout", outs_when_up=0, inning_topbot="Top"),
    ]
    return pd.DataFrame(rows)


def test_build_runner_transitions_carries_outs_column():
    transitions = build_runner_transitions(_outs_fixture())
    by_runner = transitions.set_index("runner_id")["outs"]
    assert by_runner.loc[111] == 0
    assert by_runner.loc[222] == 1
    # 333 (outs=2, stranded when the half-inning ended) never appears at all.
    assert 333 not in by_runner.index


def test_outs_2_hit_into_play_out_never_shows_as_out_due_to_stranding():
    transitions = build_runner_transitions(_outs_fixture())
    # This is the selection effect itself: both outs=0 and outs=1 runners
    # who vanish with the score unchanged are labeled OUT, but the only
    # outs=2 row that survives at all is the one that scored (444) -- 333,
    # who vanished with the score unchanged at outs=2, is dropped entirely
    # rather than labeled OUT, since this module can't tell a genuine
    # basepath out apart from ordinary stranding once the half-inning ends
    # on the same play, and refuses to guess (see module docstring).
    assert set(transitions.loc[transitions["outs"] == 0, "end_base"]) == {"OUT"}
    assert set(transitions.loc[transitions["outs"] == 1, "end_base"]) == {"OUT"}
    assert set(transitions.loc[transitions["outs"] == 2, "end_base"]) == {"HOME"}


def test_compute_league_advancement_rates_by_outs_normalizes_per_outs_group():
    transitions = build_runner_transitions(_outs_fixture())
    rates = compute_league_advancement_rates_by_outs(transitions)
    totals = rates.groupby(["outs", "start_base", "outcome"])["probability"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)
    subset = rates[(rates["outs"] == 0) & (rates["start_base"] == "3B") & (rates["outcome"] == "hit_into_play_out")]
    assert subset.set_index("end_base")["probability"].loc["OUT"] == pytest.approx(1.0)


def test_compare_advancement_rates_by_outs_ranks_by_spread_and_flags_missing_outs():
    transitions = build_runner_transitions(_outs_fixture())
    rates = compute_league_advancement_rates_by_outs(transitions)
    comparison = compare_advancement_rates_by_outs(rates)
    row = comparison[
        (comparison["start_base"] == "3B") & (comparison["outcome"] == "hit_into_play_out")
        & (comparison["end_base"] == "OUT")
    ].iloc[0]
    assert row["outs_0"] == pytest.approx(1.0)
    assert row["outs_1"] == pytest.approx(1.0)
    # outs==2 was never observed for this (start_base, outcome, end_base) --
    # the pivot must show that as missing, not silently as zero.
    assert pd.isna(row["outs_2"])


# ---------- sprint speed history ----------


def _sprint_speed_table_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"batter_id": 1, "season": 2022, "sprint_speed": 30.0},
            {"batter_id": 1, "season": 2023, "sprint_speed": 30.5},
            {"batter_id": 2, "season": 2023, "sprint_speed": 24.0},
            {"batter_id": 3, "season": 2021, "sprint_speed": 27.0},
        ]
    )


def test_sprint_speed_for_exact_season():
    history = build_sprint_speed_history(_sprint_speed_table_fixture())
    assert sprint_speed_for(history, 1, 2023) == pytest.approx(30.5)


def test_sprint_speed_for_falls_back_to_nearest_season():
    history = build_sprint_speed_history(_sprint_speed_table_fixture())
    # Player 3 only has 2021 on record -- queried for 2023, falls back to it.
    assert sprint_speed_for(history, 3, 2023) == pytest.approx(27.0)


def test_sprint_speed_for_unknown_player_returns_none():
    history = build_sprint_speed_history(_sprint_speed_table_fixture())
    assert sprint_speed_for(history, 999999, 2023) is None


def test_league_avg_sprint_speed_for_known_and_unknown_season():
    history = build_sprint_speed_history(_sprint_speed_table_fixture())
    expected_2023 = (30.5 + 24.0) / 2
    assert league_avg_sprint_speed_for(history, 2023) == pytest.approx(expected_2023)
    # A season nowhere in the table falls back to the overall average.
    assert league_avg_sprint_speed_for(history, 1999) == pytest.approx(history.overall_league_avg)


# ---------- adjust_for_sprint_speed ----------


def test_adjust_for_sprint_speed_fast_runner_shifts_toward_aggressive():
    distribution = {"2B": 0.7, "3B": 0.25, "OUT": 0.05}
    league_avg = 27.0
    fast = adjust_for_sprint_speed(distribution, runner_sprint_speed=30.0, league_avg_sprint_speed=league_avg, sensitivity=0.5)
    assert fast["3B"] > distribution["3B"]
    assert fast["2B"] < distribution["2B"]
    assert fast["OUT"] == pytest.approx(distribution["OUT"])  # P(OUT) untouched
    assert sum(fast.values()) == pytest.approx(1.0)


def test_adjust_for_sprint_speed_slow_runner_shifts_toward_baseline():
    distribution = {"2B": 0.7, "3B": 0.25, "OUT": 0.05}
    slow = adjust_for_sprint_speed(distribution, runner_sprint_speed=24.0, league_avg_sprint_speed=27.0, sensitivity=0.5)
    assert slow["3B"] < distribution["3B"]
    assert slow["2B"] > distribution["2B"]
    assert sum(slow.values()) == pytest.approx(1.0)


def test_adjust_for_sprint_speed_average_runner_is_unchanged():
    distribution = {"2B": 0.7, "3B": 0.25, "OUT": 0.05}
    result = adjust_for_sprint_speed(distribution, runner_sprint_speed=27.0, league_avg_sprint_speed=27.0, sensitivity=0.5)
    for key in distribution:
        assert result[key] == pytest.approx(distribution[key])


def test_adjust_for_sprint_speed_clips_rather_than_going_negative():
    distribution = {"2B": 0.05, "3B": 0.9, "OUT": 0.05}
    # An absurdly fast runner shouldn't be able to push "2B" probability below zero.
    result = adjust_for_sprint_speed(distribution, runner_sprint_speed=100.0, league_avg_sprint_speed=27.0, sensitivity=0.5)
    assert result["2B"] >= 0.0
    assert sum(result.values()) == pytest.approx(1.0)


def test_adjust_for_sprint_speed_single_non_out_outcome_is_unchanged():
    distribution = {"HOME": 0.95, "OUT": 0.05}
    result = adjust_for_sprint_speed(distribution, runner_sprint_speed=30.0, league_avg_sprint_speed=27.0, sensitivity=0.5)
    assert result == distribution


# ---------- BaserunningModel ----------


def _model_fixture() -> BaserunningModel:
    transitions = build_runner_transitions(_transitions_fixture())
    rates = compute_league_advancement_rates(transitions)
    sprint_speed_history = build_sprint_speed_history(_sprint_speed_table_fixture())
    return BaserunningModel(rates, BaserunningConfig(speed_adjustment_sensitivity=0.5), sprint_speed_history)


def test_baserunning_model_league_distribution_unknown_combo_is_empty():
    model = _model_fixture()
    assert model.league_distribution("2B", "triple") == {}


def test_baserunning_model_advancement_distribution_without_runner_is_league_rate():
    model = _model_fixture()
    league = model.league_distribution("1B", "single")
    assert model.advancement_distribution("1B", "single") == league


def test_baserunning_model_advancement_distribution_adjusts_for_a_known_fast_runner():
    model = _model_fixture()
    league = model.league_distribution("1B", "single")
    adjusted = model.advancement_distribution("1B", "single", runner_id=1, season=2023)  # sprint_speed=30.5, fast
    assert adjusted["3B"] > league["3B"]


def test_baserunning_model_advancement_distribution_falls_back_for_unknown_runner():
    model = _model_fixture()
    league = model.league_distribution("1B", "single")
    result = model.advancement_distribution("1B", "single", runner_id=999999, season=2023)
    assert result == league


def test_baserunning_model_probability_matches_distribution():
    model = _model_fixture()
    assert model.probability("2B", "single", "HOME") == pytest.approx(1.0)
    assert model.probability("2B", "single", "OUT") == pytest.approx(0.0)


def _outs_model_fixture() -> BaserunningModel:
    transitions = build_runner_transitions(_outs_fixture())
    rates = compute_league_advancement_rates(transitions)
    rates_by_outs = compute_league_advancement_rates_by_outs(transitions)
    return BaserunningModel(rates, BaserunningConfig(), rates_by_outs=rates_by_outs)


def test_baserunning_model_league_distribution_uses_outs_specific_slice_when_observed():
    model = _outs_model_fixture()
    assert model.league_distribution("3B", "hit_into_play_out", outs=0) == {"OUT": pytest.approx(1.0)}


def test_baserunning_model_league_distribution_falls_back_to_pooled_when_outs_slice_unobserved():
    model = _outs_model_fixture()
    # outs=99 was never observed for 3B/hit_into_play_out at all -- must
    # fall back to the (non-empty) pooled distribution rather than {}.
    pooled = model.league_distribution("3B", "hit_into_play_out")
    assert pooled != {}
    assert model.league_distribution("3B", "hit_into_play_out", outs=99) == pooled


def test_baserunning_model_league_distribution_without_outs_arg_ignores_outs_table():
    model = _outs_model_fixture()
    assert model.league_distribution("3B", "hit_into_play_out") == model.league_distribution(
        "3B", "hit_into_play_out", outs=None
    )


def _pandas_league_distribution(rates, rates_by_outs, start_base, outcome, outs=None):
    """Independent re-implementation of BaserunningModel.league_distribution's
    original pandas boolean-mask logic -- removed from the production class
    in favor of a pre-indexed dict built once in __post_init__ (profiling a
    real batched game_engine.py simulation found this pandas filter, mostly
    Arrow-string column comparisons and DataFrame reindexing rather than the
    filtering logic itself, was ~60% of total simulation wall time). Kept
    here purely as a test oracle, not production code."""
    if outs is not None and rates_by_outs is not None:
        subset = rates_by_outs[
            (rates_by_outs["outs"] == outs)
            & (rates_by_outs["start_base"] == start_base)
            & (rates_by_outs["outcome"] == outcome)
        ]
        if not subset.empty:
            return dict(zip(subset["end_base"], subset["probability"]))
    subset = rates[(rates["start_base"] == start_base) & (rates["outcome"] == outcome)]
    return dict(zip(subset["end_base"], subset["probability"]))


def test_league_distribution_dict_index_matches_direct_pandas_filtering_for_every_combination():
    # The pre-indexed dict lookup must return exactly what the original
    # pandas boolean-mask filter would have, for every real (start_base,
    # outcome) combination in both the pooled and outs-split tables, every
    # outs value actually used (0/1/2), an out count never observed
    # (exercises the pooled fallback), and outs=None (skips the outs-split
    # table entirely).
    transitions = build_runner_transitions(_outs_fixture())
    rates = compute_league_advancement_rates(transitions)
    rates_by_outs = compute_league_advancement_rates_by_outs(transitions)
    model = BaserunningModel(rates, BaserunningConfig(), rates_by_outs=rates_by_outs)

    combos = rates[["start_base", "outcome"]].drop_duplicates()
    checked = 0
    for start_base, outcome in zip(combos["start_base"], combos["outcome"]):
        for outs in (None, 0, 1, 2, 99):
            expected = _pandas_league_distribution(rates, rates_by_outs, start_base, outcome, outs)
            actual = model.league_distribution(start_base, outcome, outs)
            assert actual.keys() == expected.keys(), f"key mismatch for ({start_base}, {outcome}, outs={outs})"
            for end_base in expected:
                assert actual[end_base] == pytest.approx(expected[end_base]), (
                    f"value mismatch for ({start_base}, {outcome}, outs={outs}, end_base={end_base})"
                )
            checked += 1
    assert checked > 0  # sanity: the fixture actually exercised real combinations, not zero rows

    # An entirely unobserved (start_base, outcome) pair -- both paths must
    # agree on {}.
    assert model.league_distribution("3B", "walk") == {}
    assert _pandas_league_distribution(rates, rates_by_outs, "3B", "walk") == {}


def test_league_distribution_returns_a_fresh_dict_each_call_not_a_shared_reference():
    # The dict-index lookup must preserve the original pandas-based
    # method's observable contract: a caller mutating the returned dict
    # must never corrupt what a later call returns.
    model = _model_fixture()
    first = model.league_distribution("1B", "single")
    first["3B"] = 999.0
    second = model.league_distribution("1B", "single")
    assert second["3B"] != 999.0


# ---------- persistence ----------


def test_save_and_load_model_round_trips(tmp_path):
    model = _model_fixture()
    path = tmp_path / "baserunning.pkl"
    save_model(model, path)
    loaded = load_model(path)

    assert loaded.advancement_distribution("1B", "single", runner_id=1, season=2023) == pytest.approx(
        model.advancement_distribution("1B", "single", runner_id=1, season=2023)
    )


# ---------- main() end-to-end ----------


def test_main_runs_end_to_end_and_saves_a_checkpoint(tmp_path):
    from src.data.statcast_common import write_partitioned

    pitches_dir = tmp_path / "pitches"
    fixture = _transitions_fixture()
    fixture["is_valid"] = True
    write_partitioned(fixture, pitches_dir)

    checkpoint_path = tmp_path / "baserunning.pkl"
    # --sprint-speed-dir deliberately points at a path that doesn't exist,
    # rather than relying on the real default -- this project's own
    # data/processed/sprint_speed may genuinely exist on whatever machine
    # runs this test, which would silently pull real data into what's
    # supposed to be an isolated test.
    baserunning_main(
        [
            "--pitches-dir", str(pitches_dir),
            "--checkpoint", str(checkpoint_path),
            "--sprint-speed-dir", str(tmp_path / "no_sprint_speed_here"),
        ]
    )

    assert checkpoint_path.exists()
    model = load_model(checkpoint_path)
    assert isinstance(model, BaserunningModel)
    assert model.sprint_speed_history is None
    distribution = model.league_distribution("2B", "single")
    assert distribution.get("HOME") == pytest.approx(1.0)


def test_main_train_season_start_and_val_season_end_flags_override_the_default_split(tmp_path, caplog):
    """Walk-forward retraining needs a non-default season boundary --
    confirms --train-season-start/--val-season-end actually change how many
    pitches feed the rate table, not just that they parse."""
    from src.data.statcast_common import write_partitioned

    fixture_2023 = _transitions_fixture()
    fixture_2023["is_valid"] = True
    fixture_2015 = fixture_2023.copy()
    fixture_2015["season"] = 2015
    fixture_2015["game_date"] = pd.Timestamp("2015-04-01")
    fixture_2015["game_pk"] = fixture_2015["game_pk"] + 10000  # keep game_pk distinct across seasons

    pitches_dir = tmp_path / "pitches"
    write_partitioned(pd.concat([fixture_2015, fixture_2023], ignore_index=True), pitches_dir)

    with caplog.at_level("INFO"):
        baserunning_main([
            "--pitches-dir", str(pitches_dir),
            "--checkpoint", str(tmp_path / "baserunning.pkl"),
            "--sprint-speed-dir", str(tmp_path / "no_sprint_speed_here"),
            "--train-season-start", "2015",
            "--val-season-end", "2015",
        ])

    restricted_lines = [line for line in caplog.text.splitlines() if "Restricted to seasons" in line]
    assert len(restricted_lines) == 1
    assert "2015-2015" in restricted_lines[0]
    restricted_pitch_count = int(restricted_lines[0].split("(")[1].split(" pitches")[0])
    assert restricted_pitch_count == len(fixture_2015)  # only the 2015 half of the combined fixture


def test_main_loads_sprint_speed_data_when_the_directory_exists(tmp_path):
    from src.data.statcast_common import write_partitioned

    pitches_dir = tmp_path / "pitches"
    fixture = _transitions_fixture()
    fixture["is_valid"] = True
    write_partitioned(fixture, pitches_dir)

    # fetch_sprint_speed.py writes plain partitioned parquet directly (no
    # "is_valid" column, so write_partitioned's own logging would KeyError) --
    # match that convention here rather than this project's pitch-table one.
    sprint_speed_dir = tmp_path / "sprint_speed"
    _sprint_speed_table_fixture().to_parquet(sprint_speed_dir, partition_cols=["season"], index=False)

    checkpoint_path = tmp_path / "baserunning.pkl"
    baserunning_main(
        [
            "--pitches-dir", str(pitches_dir),
            "--checkpoint", str(checkpoint_path),
            "--sprint-speed-dir", str(sprint_speed_dir),
        ]
    )

    model = load_model(checkpoint_path)
    assert model.sprint_speed_history is not None
    assert sprint_speed_for(model.sprint_speed_history, 1, 2023) == pytest.approx(30.5)
