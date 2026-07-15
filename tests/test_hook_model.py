import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data.statcast_common import TRAIN_SEASON_RANGE, VAL_SEASONS, write_partitioned
from src.models.hook_model import (
    HOOK_FEATURE_NAMES,
    PITCH_COUNT_MILESTONE_FEATURE_NAMES,
    PITCH_COUNT_MILESTONES,
    STARTER_FEATURE_NAMES,
    HookModelPredictor,
    attach_pitch_count_milestones,
    attach_removal_history_features,
    attach_removal_history_features_by_role,
    backtest_removal_point,
    build_hook_examples,
    build_removal_history,
    classify_game_closeness,
    evaluate_hazard_predictions,
    expected_removal_point,
    fit_and_compare_role_specific_models,
    fit_hook_model,
    load_predictor,
    main as hook_main,
    predict_hazard,
    removal_history_features_for,
    save_predictor,
    summarize_removal_point_errors,
    summarize_removal_point_errors_by_group,
)


def _at_bat_rows(game_pk, at_bat_number, n_pitches, inning, inning_topbot, pitcher_id, batter_id,
                  home_score=0, away_score=0, on_1b=None, on_2b=None, on_3b=None,
                  game_date="2023-04-01", season=2023, home_team="DET", away_team="CLE", times_through_order=0):
    """One row per actual pitch (pitch_number 1..n_pitches) -- build_hook_examples
    counts *rows* per at-bat to get the pitch count, matching the real
    per-pitch grain of the processed pitches table, so a fixture needs one
    row per pitch, not one row with pitch_number set to a pitch total."""
    return [
        {
            "game_pk": game_pk, "at_bat_number": at_bat_number, "pitch_number": pitch_number,
            "inning": inning, "inning_topbot": inning_topbot, "pitcher_id": pitcher_id, "batter_id": batter_id,
            "home_team": home_team, "away_team": away_team, "home_score": home_score, "away_score": away_score,
            "on_1b": on_1b, "on_2b": on_2b, "on_3b": on_3b, "times_through_order": times_through_order,
            "game_date": pd.Timestamp(game_date), "season": season,
        }
        for pitch_number in range(1, n_pitches + 1)
    ]


def _game_fixture() -> pd.DataFrame:
    """One DET-home-vs-CLE-away game, 3 innings:
    - DET pitching (Top half): pitcher 100 faces b1, b2, b3 (11 pitches
      total) then is pulled for pitcher 200, who faces b4, b5 (8 pitches)
      then is pulled for pitcher 300, who faces only b6 -- DET's last
      pitching stint of the game, so b6's row is censored (dropped).
    - CLE pitching (Bottom half): pitcher 900 pitches the entire game
      without ever being substituted (b10, b11, b12, b13, b14) -- CLE's
      only (and therefore also last) stint, so only its final row (b14) is
      censored; the four rows before it are genuine "not removed" negatives.

    A runner is on 1st as of at_bat 2's first pitch (i.e. the state right
    after at_bat 1 concludes) to exercise the runner_on_base feature, and
    the score changes after at_bat 4 to exercise run_differential.
    """
    rows = []
    rows += _at_bat_rows(1, 1, 4, 1, "Top", 100, 1)  # DET stint 1, batter 1 (b1)
    rows += _at_bat_rows(1, 2, 3, 1, "Top", 100, 2, on_1b=555)  # b2 -- runner on 1st as of THIS at-bat's start
    rows += _at_bat_rows(1, 3, 5, 1, "Bot", 900, 10)  # CLE stint, b10
    rows += _at_bat_rows(1, 4, 4, 1, "Bot", 900, 11, home_score=1)  # b11 -- DET now leads 1-0 as of this at-bat's start
    rows += _at_bat_rows(1, 5, 4, 2, "Top", 100, 3)  # DET stint 1, b3 -- last batter for pitcher 100
    rows += _at_bat_rows(1, 6, 3, 2, "Top", 200, 4)  # DET stint 2 (pitcher 200), b4
    rows += _at_bat_rows(1, 7, 3, 2, "Bot", 900, 12)  # CLE stint continues, b12
    rows += _at_bat_rows(1, 8, 4, 2, "Bot", 900, 13)  # CLE stint continues, b13
    rows += _at_bat_rows(1, 9, 5, 3, "Top", 200, 5)  # DET stint 2, b5 -- last batter for pitcher 200
    rows += _at_bat_rows(1, 10, 4, 3, "Top", 300, 6)  # DET stint 3 (pitcher 300), b6 -- DET's last stint (censored)
    rows += _at_bat_rows(1, 11, 3, 3, "Bot", 900, 14)  # CLE stint continues, b14 -- CLE's last batter (censored)
    return pd.DataFrame(rows)


# ---------- build_hook_examples ----------


def test_build_hook_examples_handles_nullable_int64_columns():
    """The real processed pitches table stores game_pk/at_bat_number/
    pitcher_id/batter_id as pandas nullable Int64 -- comparing a
    .shift()-introduced NA against a nullable column produces pd.NA (not
    True/False), which silently poisons every downstream boolean op and
    used to crash the final label .astype(int). Regression check with the
    fixture cast to the real dtype."""
    game = _game_fixture()
    for col in ("game_pk", "at_bat_number", "pitcher_id", "batter_id", "home_score", "away_score", "times_through_order"):
        game[col] = game[col].astype("Int64")
    examples = build_hook_examples(game)
    assert len(examples) == 9


def test_build_hook_examples_row_count_and_labels():
    examples = build_hook_examples(_game_fixture())
    # pitcher 100 (3 rows), pitcher 200 (2 rows), pitcher 900 (4 rows, last dropped) = 9.
    # pitcher 300's single row (censored) is dropped entirely.
    assert len(examples) == 9
    assert set(examples["pitcher_id"].unique()) == {100, 200, 900}

    pitcher100 = examples[examples["pitcher_id"] == 100].sort_values("batters_faced_so_far")
    assert pitcher100["label"].tolist() == [0, 0, 1]
    assert pitcher100["batters_faced_so_far"].tolist() == [1, 2, 3]
    assert pitcher100["pitch_count"].tolist() == [4, 7, 11]

    pitcher200 = examples[examples["pitcher_id"] == 200].sort_values("batters_faced_so_far")
    assert pitcher200["label"].tolist() == [0, 1]
    assert pitcher200["pitch_count"].tolist() == [3, 8]

    pitcher900 = examples[examples["pitcher_id"] == 900].sort_values("batters_faced_so_far")
    assert pitcher900["label"].tolist() == [0, 0, 0, 0]  # only 4 of its 5 real batters survive censoring
    assert pitcher900["batters_faced_so_far"].tolist() == [1, 2, 3, 4]


def test_build_hook_examples_ignores_half_inning_switches_as_substitutions():
    """Pitcher 100 pitches across innings 1 and 2 (Top halves) without ever
    being substituted -- the intervening Bottom-half at-bats (a different
    team's pitcher) must not be mistaken for a hook."""
    examples = build_hook_examples(_game_fixture())
    pitcher100 = examples[examples["pitcher_id"] == 100]
    assert (pitcher100["stint_id"] == pitcher100["stint_id"].iloc[0]).all()


def test_build_hook_examples_runner_on_base_reflects_the_next_at_bat():
    examples = build_hook_examples(_game_fixture())
    # b1's row (pitcher 100, batters_faced_so_far=1): the *next* at-bat (b2)
    # starts with a runner on 1st.
    row = examples[(examples["pitcher_id"] == 100) & (examples["batters_faced_so_far"] == 1)].iloc[0]
    assert row["runner_on_base"] == 1.0
    # b2's row: the next at-bat (b10, CLE's first) has no runners on.
    row2 = examples[(examples["pitcher_id"] == 100) & (examples["batters_faced_so_far"] == 2)].iloc[0]
    assert row2["runner_on_base"] == 0.0


def test_build_hook_examples_run_differential_uses_the_pitchers_own_team_perspective():
    examples = build_hook_examples(_game_fixture())
    # b10's row (pitcher 900, CLE, batters_faced_so_far=1): the next at-bat
    # (b11) starts with home_score=1, away_score=0 -- DET (home) leads by 1.
    # From CLE (pitcher 900's team, away)'s own perspective, that's -1.
    row = examples[(examples["pitcher_id"] == 900) & (examples["batters_faced_so_far"] == 1)].iloc[0]
    assert row["run_differential"] == pytest.approx(-1.0)


def test_build_hook_examples_drops_censored_final_stints():
    examples = build_hook_examples(_game_fixture())
    assert 300 not in examples["pitcher_id"].unique()  # DET's last stint, fully censored
    # CLE's stint (pitcher 900) keeps its first 4 batters but not the 5th (b14).
    assert len(examples[examples["pitcher_id"] == 900]) == 4


def test_build_hook_examples_is_starter_flags_only_each_teams_first_stint():
    examples = build_hook_examples(_game_fixture())
    # DET: pitcher 100 started, pitcher 200 relieved (pitcher 300's censored stint was dropped).
    assert examples[examples["pitcher_id"] == 100]["is_starter"].all()
    assert not examples[examples["pitcher_id"] == 200]["is_starter"].any()
    # CLE: pitcher 900 was its only (and therefore starting) pitcher all game.
    assert examples[examples["pitcher_id"] == 900]["is_starter"].all()


# ---------- removal history ----------


def test_removal_history_features_for_unknown_pitcher_falls_back_to_league_average():
    examples = build_hook_examples(_game_fixture())
    history = build_removal_history(examples, league_avg_examples=examples)
    avg_batters, avg_pitches = removal_history_features_for(history, 999999, pd.Timestamp("2023-05-01").value)
    completed = examples[examples["label"] == 1]
    assert avg_batters == pytest.approx(completed["batters_faced_so_far"].mean())
    assert avg_pitches == pytest.approx(completed["pitch_count"].mean())


def test_removal_history_features_for_uses_only_strictly_prior_completed_stints():
    examples = pd.DataFrame(
        [
            {"pitcher_id": 100, "game_date": pd.Timestamp("2023-04-01"), "batters_faced_so_far": 3, "pitch_count": 40, "label": 1},
            {"pitcher_id": 100, "game_date": pd.Timestamp("2023-04-10"), "batters_faced_so_far": 5, "pitch_count": 60, "label": 1},
        ]
    )
    history = build_removal_history(examples, league_avg_examples=examples)

    # As of the 04-10 game itself, only the 04-01 stint is strictly prior.
    avg_batters, avg_pitches = removal_history_features_for(history, 100, pd.Timestamp("2023-04-10").value)
    assert avg_batters == pytest.approx(3.0)
    assert avg_pitches == pytest.approx(40.0)

    # As of a later date, both prior stints are averaged.
    avg_batters2, avg_pitches2 = removal_history_features_for(history, 100, pd.Timestamp("2023-05-01").value)
    assert avg_batters2 == pytest.approx(4.0)
    assert avg_pitches2 == pytest.approx(50.0)


def test_attach_removal_history_features_gives_every_row_of_a_stint_the_same_values():
    examples = build_hook_examples(_game_fixture())
    history = build_removal_history(examples, league_avg_examples=examples)
    result = attach_removal_history_features(examples, history)

    pitcher100_rows = result[result["pitcher_id"] == 100]
    assert pitcher100_rows["historical_avg_batters_faced_at_removal"].nunique() == 1
    assert set(HOOK_FEATURE_NAMES) <= set(result.columns)


# ---------- fit_hook_model / predict_hazard / evaluate_hazard_predictions ----------


def _fitted_model_and_examples():
    examples = build_hook_examples(_game_fixture())
    history = build_removal_history(examples, league_avg_examples=examples)
    examples = attach_removal_history_features(examples, history)
    model = fit_hook_model(examples)
    return model, examples, history


def test_fit_hook_model_returns_a_fitted_logistic_regression():
    model, examples, _ = _fitted_model_and_examples()
    assert isinstance(model, LogisticRegression)
    probs = predict_hazard(model, examples)
    assert probs.shape == (len(examples),)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_evaluate_hazard_predictions_perfect_separation():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])
    metrics = evaluate_hazard_predictions(y_true, y_prob)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)


# ---------- expected_removal_point / backtest_removal_point ----------


def test_expected_removal_point_is_exact_when_hazard_is_deterministic():
    """A hand-built 'model' (via a stub with a fixed predict_proba) that
    says hazard=1.0 at the true removal batter and 0.0 before it should
    predict the removal point exactly."""

    class _StubModel:
        def predict_proba(self, X):
            # X columns follow HOOK_FEATURE_NAMES order; batters_faced_so_far is column 0.
            batters = X[:, 0]
            hazard = (batters == batters.max()).astype(float)
            return np.column_stack([1 - hazard, hazard])

    stint = pd.DataFrame(
        {
            "batters_faced_so_far": [1, 2, 3],
            "pitch_count": [10, 20, 30],
            "run_differential": [0, 0, 0],
            "runner_on_base": [0, 0, 0],
            "historical_avg_batters_faced_at_removal": [3, 3, 3],
            "historical_avg_pitch_count_at_removal": [30, 30, 30],
        }
    )
    expected_batters, expected_pitches = expected_removal_point(_StubModel(), stint)
    assert expected_batters == pytest.approx(3.0)
    assert expected_pitches == pytest.approx(30.0)


def test_backtest_removal_point_actual_matches_the_stints_own_last_row():
    model, examples, _ = _fitted_model_and_examples()
    results = backtest_removal_point(model, examples)
    assert len(results) == examples["stint_id"].nunique()
    for _, row in results.iterrows():
        stint_rows = examples[examples["stint_id"] == row["stint_id"]]
        assert row["actual_batters_faced"] == stint_rows["batters_faced_so_far"].max()
        assert row["actual_pitch_count"] == stint_rows["pitch_count"].max()


def test_backtest_removal_point_carries_is_starter_and_run_differential_at_removal():
    model, examples, _ = _fitted_model_and_examples()
    results = backtest_removal_point(model, examples).set_index("pitcher_id")

    assert results.loc[100, "is_starter"] == True  # noqa: E712
    assert results.loc[200, "is_starter"] == False  # noqa: E712

    # Pitcher 100's removal point (b3, batters_faced_so_far=3): the actual
    # removal-point run_differential is that row's own run_differential.
    stint_rows = examples[examples["pitcher_id"] == 100].sort_values("batters_faced_so_far")
    expected_run_diff = stint_rows["run_differential"].iloc[-1]
    assert results.loc[100, "run_differential_at_removal"] == pytest.approx(expected_run_diff)


def test_summarize_removal_point_errors_matches_hand_computed_mae():
    results = pd.DataFrame(
        {
            "batters_faced_error": [1.0, -2.0, 3.0],
            "pitch_count_error": [5.0, -5.0, 0.0],
        }
    )
    summary = summarize_removal_point_errors(results)
    assert summary["n_stints"] == 3
    assert summary["batters_faced_mae"] == pytest.approx((1 + 2 + 3) / 3)
    assert summary["batters_faced_bias"] == pytest.approx((1 - 2 + 3) / 3)
    assert summary["pitch_count_mae"] == pytest.approx((5 + 5 + 0) / 3)


# ---------- classify_game_closeness / summarize_removal_point_errors_by_group ----------


def test_classify_game_closeness_uses_the_abs_run_differential_threshold():
    run_diff = pd.Series([0, 2, -2, 3, -5])
    result = classify_game_closeness(run_diff)
    assert result.tolist() == ["close", "close", "close", "blowout", "blowout"]


def test_summarize_removal_point_errors_by_group_splits_correctly():
    results = pd.DataFrame(
        {
            "role": ["starter", "starter", "reliever", "reliever"],
            "batters_faced_error": [2.0, 4.0, -1.0, -1.0],
            "pitch_count_error": [10.0, 10.0, -2.0, -4.0],
        }
    )
    breakdown = summarize_removal_point_errors_by_group(results, "role").set_index("role")

    assert breakdown.loc["starter", "n_stints"] == 2
    assert breakdown.loc["starter", "batters_faced_mae"] == pytest.approx(3.0)
    assert breakdown.loc["starter", "batters_faced_bias"] == pytest.approx(3.0)
    assert breakdown.loc["reliever", "n_stints"] == 2
    assert breakdown.loc["reliever", "batters_faced_mae"] == pytest.approx(1.0)
    assert breakdown.loc["reliever", "batters_faced_bias"] == pytest.approx(-1.0)
    assert breakdown.loc["reliever", "pitch_count_mae"] == pytest.approx(3.0)


def test_summarize_removal_point_errors_by_group_detects_a_concentrated_bias():
    """If one group's bias is much larger than another's, the breakdown
    must actually show that difference, not average it away."""
    results = pd.DataFrame(
        {
            "game_closeness": ["close"] * 3 + ["blowout"] * 3,
            "batters_faced_error": [-5.0, -5.0, -5.0, 0.1, -0.1, 0.0],
            "pitch_count_error": [-10.0] * 3 + [0.0] * 3,
        }
    )
    breakdown = summarize_removal_point_errors_by_group(results, "game_closeness").set_index("game_closeness")
    assert breakdown.loc["close", "batters_faced_mae"] > 4 * breakdown.loc["blowout", "batters_faced_mae"]


# ---------- attach_pitch_count_milestones / attach_removal_history_features_by_role ----------


def test_attach_pitch_count_milestones_flags_thresholds_crossed():
    examples = pd.DataFrame({"pitch_count": [50, 75, 99, 100, 121]})
    result = attach_pitch_count_milestones(examples)
    assert result["pitch_count_ge_75"].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]
    assert result["pitch_count_ge_100"].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert result["pitch_count_ge_120"].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_attach_removal_history_features_by_role_uses_the_matching_role_history():
    starter_examples = pd.DataFrame(
        [{"pitcher_id": 100, "game_date": pd.Timestamp("2023-04-01"), "batters_faced_so_far": 20, "pitch_count": 90, "label": 1}]
    )
    reliever_examples = pd.DataFrame(
        [{"pitcher_id": 100, "game_date": pd.Timestamp("2023-04-01"), "batters_faced_so_far": 3, "pitch_count": 15, "label": 1}]
    )
    starter_history = build_removal_history(starter_examples, league_avg_examples=starter_examples)
    reliever_history = build_removal_history(reliever_examples, league_avg_examples=reliever_examples)

    # Same pitcher, same date, but a different role per row -- a swingman
    # scenario for exercising that the merge key really is (pitcher_id,
    # game_date, is_starter), not just (pitcher_id, game_date).
    examples = pd.DataFrame(
        [
            {"pitcher_id": 100, "game_date": pd.Timestamp("2023-04-10"), "is_starter": True},
            {"pitcher_id": 100, "game_date": pd.Timestamp("2023-04-10"), "is_starter": False},
        ]
    )
    result = attach_removal_history_features_by_role(examples, starter_history, reliever_history)
    starter_row = result[result["is_starter"]].iloc[0]
    reliever_row = result[~result["is_starter"]].iloc[0]
    assert starter_row["historical_avg_batters_faced_at_removal"] == pytest.approx(20.0)
    assert reliever_row["historical_avg_batters_faced_at_removal"] == pytest.approx(3.0)


# ---------- fit_and_compare_role_specific_models ----------


def test_fit_and_compare_role_specific_models_returns_a_row_per_model_role_combo():
    examples = build_hook_examples(_game_fixture())
    starter_model, reliever_model, starter_history, reliever_history, comparison = fit_and_compare_role_specific_models(
        examples, examples, examples
    )
    assert isinstance(starter_model, LogisticRegression)
    assert isinstance(reliever_model, LogisticRegression)  # this is the hybrid model -- see fit_and_compare_role_specific_models
    assert set(comparison["model"]) == {
        "pooled_on_starters", "starter_model_on_starters", "pooled_on_relievers",
        "reliever_only_model_on_relievers", "hybrid_reliever_model_on_relievers",
    }
    assert {"n_stints", "batters_faced_mae", "batters_faced_bias", "pitch_count_mae", "pitch_count_bias"} <= set(comparison.columns)


# ---------- HookModelPredictor + persistence ----------


def _role_fitted_predictor_and_examples():
    examples = build_hook_examples(_game_fixture())
    starter_model, reliever_model, starter_history, reliever_history, _ = fit_and_compare_role_specific_models(
        examples, examples, examples
    )
    return HookModelPredictor(starter_model, reliever_model, starter_history, reliever_history), examples


def test_predictor_predict_proba_matches_manual_feature_row_for_reliever():
    """The "reliever" slot now holds the hybrid model, which expects
    STARTER_FEATURE_NAMES (the expanded feature set) just like the starter
    model does -- see HookModelPredictor's docstring."""
    predictor, examples = _role_fitted_predictor_and_examples()

    single = predictor.predict_proba(
        pitcher_id=200, as_of_date="2023-04-01", is_starter=False,
        batters_faced_so_far=1, pitch_count=3, run_differential=0.0, runner_on_base=False,
    )
    avg_batters, avg_pitches = removal_history_features_for(predictor.reliever_history, 200, pd.Timestamp("2023-04-01").value)
    milestones = {name: float(3 >= m) for name, m in zip(PITCH_COUNT_MILESTONE_FEATURE_NAMES, PITCH_COUNT_MILESTONES)}
    row = pd.DataFrame(
        [{"batters_faced_so_far": 1, "pitch_count": 3, "run_differential": 0.0, "runner_on_base": 0.0,
          "historical_avg_batters_faced_at_removal": avg_batters, "historical_avg_pitch_count_at_removal": avg_pitches,
          **milestones, "times_through_order": 0.0}]
    )
    expected = predict_hazard(predictor.reliever_model, row, STARTER_FEATURE_NAMES)[0]
    assert single == pytest.approx(expected)


def test_predictor_predict_proba_matches_manual_feature_row_for_starter():
    predictor, examples = _role_fitted_predictor_and_examples()

    single = predictor.predict_proba(
        pitcher_id=100, as_of_date="2023-04-01", is_starter=True,
        batters_faced_so_far=2, pitch_count=95, run_differential=1.0, runner_on_base=True, times_through_order=1,
    )
    avg_batters, avg_pitches = removal_history_features_for(predictor.starter_history, 100, pd.Timestamp("2023-04-01").value)
    milestones = {name: float(95 >= m) for name, m in zip(PITCH_COUNT_MILESTONE_FEATURE_NAMES, PITCH_COUNT_MILESTONES)}
    row = pd.DataFrame(
        [{"batters_faced_so_far": 2, "pitch_count": 95, "run_differential": 1.0, "runner_on_base": 1.0,
          "historical_avg_batters_faced_at_removal": avg_batters, "historical_avg_pitch_count_at_removal": avg_pitches,
          **milestones, "times_through_order": 1.0}]
    )
    expected = predict_hazard(predictor.starter_model, row, STARTER_FEATURE_NAMES)[0]
    assert single == pytest.approx(expected)


def test_predictor_predict_proba_batch_dispatches_by_role():
    predictor, examples = _role_fitted_predictor_and_examples()
    role_examples = attach_removal_history_features_by_role(examples, predictor.starter_history, predictor.reliever_history)
    role_examples = attach_pitch_count_milestones(role_examples)

    batch = predictor.predict_proba_batch(role_examples)

    is_starter = role_examples["is_starter"].to_numpy()
    expected = np.empty(len(role_examples))
    expected[is_starter] = predict_hazard(predictor.starter_model, role_examples.loc[is_starter], STARTER_FEATURE_NAMES)
    expected[~is_starter] = predict_hazard(predictor.reliever_model, role_examples.loc[~is_starter], STARTER_FEATURE_NAMES)
    assert np.allclose(batch, expected)


def test_save_and_load_predictor_round_trips(tmp_path):
    predictor, examples = _role_fitted_predictor_and_examples()

    path = tmp_path / "hook_model.pkl"
    save_predictor(predictor, path)
    loaded = load_predictor(path)

    single_original = predictor.predict_proba(100, "2023-04-01", True, 2, 11, 0.0, False)
    single_loaded = loaded.predict_proba(100, "2023-04-01", True, 2, 11, 0.0, False)
    assert single_loaded == pytest.approx(single_original)


# ---------- main() end-to-end ----------


def _write_multi_season_fixture(pitches_dir):
    """A handful of games per season (2015-2023), each shaped like
    _game_fixture (distinct pitcher_ids per season so removal-history
    lookups have real per-pitcher data to draw on across seasons)."""
    all_rows = []
    seasons = list(range(TRAIN_SEASON_RANGE[0], TRAIN_SEASON_RANGE[1] + 1)) + list(VAL_SEASONS)
    for season in seasons:
        for game_idx in range(3):
            game_pk = season * 10 + game_idx
            date = f"{season}-04-{1 + game_idx:02d}"
            game = _game_fixture()
            game["game_pk"] = game_pk
            game["game_date"] = pd.Timestamp(date)
            game["season"] = season
            all_rows.append(game)

    pitches = pd.concat(all_rows, ignore_index=True)
    pitches["is_valid"] = True
    write_partitioned(pitches, pitches_dir)


def test_main_runs_end_to_end_and_saves_a_checkpoint(tmp_path):
    pitches_dir = tmp_path / "pitches"
    _write_multi_season_fixture(pitches_dir)

    checkpoint_path = tmp_path / "hook_model.pkl"
    hook_main(["--pitches-dir", str(pitches_dir), "--checkpoint", str(checkpoint_path)])

    assert checkpoint_path.exists()
    predictor = load_predictor(checkpoint_path)
    assert isinstance(predictor.starter_model, LogisticRegression)
    assert isinstance(predictor.reliever_model, LogisticRegression)
    prob = predictor.predict_proba(100, "2023-04-01", True, 2, 7, 0.0, False)
    assert 0.0 <= prob <= 1.0
