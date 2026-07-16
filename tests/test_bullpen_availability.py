import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data.build_features import build_season_pitches_from_frame
from src.data.statcast_common import (
    TRAIN_SEASON_RANGE,
    VAL_SEASONS,
    build_pitch_frame_from_raw,
    write_partitioned,
)
from src.models.bullpen_availability import (
    CLOSER_ONLY_FEATURE_NAMES,
    CLOSER_RECENCY_FEATURE_NAMES,
    TEAM_SAVE_OPPORTUNITY_FEATURE_NAME,
    WORKLOAD_FEATURE_NAMES,
    BullpenAvailabilityConfig,
    BullpenAvailabilityPredictor,
    HeuristicAvailabilityModel,
    ModelSelectionResult,
    RoleAwareLogisticRegressionModel,
    attach_closer_recency_features,
    attach_roles,
    attach_team_save_opportunity_feature,
    build_query_examples,
    build_team_save_opportunity_history,
    build_workload_history,
    calibration_by_role,
    classify_reliever_roles,
    closer_recency_features_for,
    compute_calibration_table,
    compute_entry_innings,
    compute_entry_situations,
    compute_pitch_counts,
    compute_team_save_opportunity_counts,
    evaluate_predictions,
    fit_logistic_regression_model,
    load_predictor,
    main as bullpen_main,
    mean_abs_calibration_gap,
    plot_calibration_reliability,
    save_predictor,
    select_availability_model,
    team_save_opportunities_trailing_for,
    workload_features_for,
)


def _pitcher_appearances_fixture() -> pd.DataFrame:
    """Team DET, one starter + two relievers across 3 games:
    - game 0 (2023-03-25): reliever 201 appears (is_starter=False), no
      designated starter recorded for this game (irrelevant to the fixture).
    - game 1 (2023-04-01): starter 100, reliever 200 relieves.
    - game 2 (2023-04-02): starter 101, reliever 200 relieves again
      (back-to-back for 200); reliever 201's last appearance was 8 days ago
      (game 0), well outside the 3/7-day feature windows but still inside
      the 14-day candidate window; former-starter 100 is also a candidate
      (only *that game's own* starter is excluded, not past starters) but
      doesn't appear in game 2 -> negative label.
    """
    return pd.DataFrame(
        [
            {"game_pk": 0, "team": "DET", "pitcher_id": 201, "game_date": pd.Timestamp("2023-03-25"), "season": 2023, "is_starter": False},
            {"game_pk": 1, "team": "DET", "pitcher_id": 100, "game_date": pd.Timestamp("2023-04-01"), "season": 2023, "is_starter": True},
            {"game_pk": 1, "team": "DET", "pitcher_id": 200, "game_date": pd.Timestamp("2023-04-01"), "season": 2023, "is_starter": False},
            {"game_pk": 2, "team": "DET", "pitcher_id": 101, "game_date": pd.Timestamp("2023-04-02"), "season": 2023, "is_starter": True},
            {"game_pk": 2, "team": "DET", "pitcher_id": 200, "game_date": pd.Timestamp("2023-04-02"), "season": 2023, "is_starter": False},
        ]
    )


def _pitch_counts_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_pk": 0, "pitcher_id": 201, "pitches": 25},
            {"game_pk": 1, "pitcher_id": 100, "pitches": 90},
            {"game_pk": 1, "pitcher_id": 200, "pitches": 15},
            {"game_pk": 2, "pitcher_id": 101, "pitches": 95},
            {"game_pk": 2, "pitcher_id": 200, "pitches": 20},
        ]
    )


def _history_fixture():
    return build_workload_history(_pitcher_appearances_fixture(), _pitch_counts_fixture())


# ---------- compute_pitch_counts / build_workload_history ----------


def test_compute_pitch_counts_groups_by_game_and_pitcher():
    pitches = pd.DataFrame(
        {
            "game_pk": [1, 1, 1, 2],
            "pitcher_id": [100, 100, 200, 100],
        }
    )
    counts = compute_pitch_counts(pitches)
    counts = counts.set_index(["game_pk", "pitcher_id"])["pitches"]
    assert counts.loc[(1, 100)] == 2
    assert counts.loc[(1, 200)] == 1
    assert counts.loc[(2, 100)] == 1


def test_build_workload_history_sorts_dates_ascending_per_pitcher():
    history = _history_fixture()
    dates = history.dates_by_pitcher[200]
    assert list(dates) == sorted(dates)
    assert len(dates) == 2


def test_build_workload_history_without_entry_situations_has_no_save_dates():
    """Light-appearance dates only need pitch counts (always available), so
    they're populated regardless; save-situation dates need entry_situations
    (score/inning context) and are empty without it."""
    history = _history_fixture()
    assert len(history.save_dates_by_pitcher.get(200, [])) == 0


def test_build_workload_history_populates_save_and_light_dates_from_entry_situations():
    appearances = pd.DataFrame(
        [
            {"game_pk": 1, "team": "DET", "pitcher_id": 300, "game_date": pd.Timestamp("2023-04-01"), "season": 2023, "is_starter": False},
            {"game_pk": 2, "team": "DET", "pitcher_id": 300, "game_date": pd.Timestamp("2023-04-05"), "season": 2023, "is_starter": False},
        ]
    )
    # Game 1: a save situation, heavy pitch count (not light).
    # Game 2: not a save situation, light pitch count.
    counts = pd.DataFrame([{"game_pk": 1, "pitcher_id": 300, "pitches": 35}, {"game_pk": 2, "pitcher_id": 300, "pitches": 10}])
    entry_situations = pd.DataFrame(
        [
            {"game_pk": 1, "pitcher_id": 300, "entry_inning": 9, "is_save_situation": True},
            {"game_pk": 2, "pitcher_id": 300, "entry_inning": 7, "is_save_situation": False},
        ]
    )
    history = build_workload_history(appearances, counts, entry_situations)

    assert list(history.save_dates_by_pitcher[300]) == [pd.Timestamp("2023-04-01").value]
    assert list(history.light_dates_by_pitcher[300]) == [pd.Timestamp("2023-04-05").value]


def test_build_workload_history_handles_an_appearance_missing_from_entry_situations():
    """A real (though shouldn't-happen) gap: an appearance with no matching
    entry_situations row at all -- the left merge introduces a NaN that
    must resolve to "not a save situation," not blow up the boolean-array
    indexing build_workload_history uses to select save dates."""
    appearances = pd.DataFrame(
        [
            {"game_pk": 1, "team": "DET", "pitcher_id": 300, "game_date": pd.Timestamp("2023-04-01"), "season": 2023, "is_starter": False},
            {"game_pk": 2, "team": "DET", "pitcher_id": 300, "game_date": pd.Timestamp("2023-04-05"), "season": 2023, "is_starter": False},
        ]
    )
    counts = pd.DataFrame([{"game_pk": 1, "pitcher_id": 300, "pitches": 10}, {"game_pk": 2, "pitcher_id": 300, "pitches": 10}])
    # Only game 1 has an entry_situations row -- game 2 is the gap.
    entry_situations = pd.DataFrame([{"game_pk": 1, "pitcher_id": 300, "entry_inning": 9, "is_save_situation": True}])

    history = build_workload_history(appearances, counts, entry_situations)
    assert list(history.save_dates_by_pitcher[300]) == [pd.Timestamp("2023-04-01").value]


# ---------- compute_entry_situations ----------


def _pitch_row(game_pk, pitcher_id, inning, at_bat_number, pitch_number, inning_topbot, home_score, away_score):
    return {
        "game_pk": game_pk, "pitcher_id": pitcher_id, "inning": inning,
        "at_bat_number": at_bat_number, "pitch_number": pitch_number,
        "inning_topbot": inning_topbot, "home_score": home_score, "away_score": away_score,
    }


def test_compute_entry_situations_flags_a_real_save_situation():
    # Pitcher 300 enters the 9th (Top -> DET/home is fielding) leading 3-1 (lead=2): a save situation.
    pitches = pd.DataFrame([_pitch_row(1, 300, 9, 1, 1, "Top", home_score=3, away_score=1)])
    result = compute_entry_situations(pitches).set_index("pitcher_id")
    assert result.loc[300, "entry_inning"] == 9
    assert result.loc[300, "is_save_situation"] == True  # noqa: E712


def test_compute_entry_situations_rejects_too_early_an_inning():
    pitches = pd.DataFrame([_pitch_row(1, 300, 7, 1, 1, "Top", home_score=3, away_score=1)])
    result = compute_entry_situations(pitches).set_index("pitcher_id")
    assert result.loc[300, "is_save_situation"] == False  # noqa: E712


def test_compute_entry_situations_rejects_too_large_a_lead():
    # 9th inning but a 5-run lead is a blowout, not a save situation.
    pitches = pd.DataFrame([_pitch_row(1, 300, 9, 1, 1, "Top", home_score=6, away_score=1)])
    result = compute_entry_situations(pitches).set_index("pitcher_id")
    assert result.loc[300, "is_save_situation"] == False  # noqa: E712


def test_compute_entry_situations_uses_the_first_pitch_of_the_appearance():
    # Two pitches for the same appearance -- the first (lower at_bat_number)
    # is the 9th with a save-eligible lead; a later, unrelated row shouldn't matter.
    pitches = pd.DataFrame(
        [
            _pitch_row(1, 300, 9, 1, 1, "Top", home_score=3, away_score=1),
            _pitch_row(1, 300, 9, 2, 1, "Top", home_score=3, away_score=1),
        ]
    )
    result = compute_entry_situations(pitches).set_index("pitcher_id")
    assert len(result) == 1
    assert result.loc[300, "is_save_situation"] == True  # noqa: E712


def test_compute_entry_innings_takes_the_min_inning_per_game_and_pitcher():
    pitches = pd.DataFrame(
        {
            "game_pk": [1, 1, 1, 2],
            "pitcher_id": [100, 100, 100, 100],
            "inning": [7, 8, 9, 3],
            "at_bat_number": [1, 2, 3, 1],
            "pitch_number": [1, 1, 1, 1],
            "inning_topbot": ["Top"] * 4,
            "home_score": [0] * 4,
            "away_score": [0] * 4,
        }
    )
    entry_innings = compute_entry_innings(pitches).set_index(["game_pk", "pitcher_id"])["entry_inning"]
    assert entry_innings.loc[(1, 100)] == 7
    assert entry_innings.loc[(2, 100)] == 3


# ---------- workload_features_for / closer_recency_features_for ----------


def test_workload_features_for_unknown_pitcher_returns_sentinel():
    history = _history_fixture()
    features = workload_features_for(history, pitcher_id=999999, cutoff_ns=pd.Timestamp("2023-04-02").value)
    assert features[WORKLOAD_FEATURE_NAMES.index("days_since_last_appearance")] == 365.0
    assert features[WORKLOAD_FEATURE_NAMES.index("pitches_last_appearance")] == 0.0


def test_workload_features_for_pitcher_200_as_of_game_2():
    history = _history_fixture()
    cutoff_ns = pd.Timestamp("2023-04-02").value
    features = dict(zip(WORKLOAD_FEATURE_NAMES, workload_features_for(history, 200, cutoff_ns)))

    assert features["days_since_last_appearance"] == pytest.approx(1.0)
    assert features["pitches_last_appearance"] == 15.0
    assert features["pitches_trailing_short"] == 15.0  # only the 04-01 outing is within 3 days
    assert features["pitches_trailing_long"] == 15.0  # also the only outing within 7 days
    assert features["appearances_trailing_long"] == 1.0
    assert features["back_to_back"] == 0.0  # only one prior appearance, not two consecutive days


def test_workload_features_for_pitcher_201_as_of_game_2_is_stale():
    """201's only appearance (game 0) is 8 days before game 2's cutoff --
    outside both the 3- and 7-day feature windows, even though it's still
    inside the 14-day candidate-pool window."""
    history = _history_fixture()
    cutoff_ns = pd.Timestamp("2023-04-02").value
    features = dict(zip(WORKLOAD_FEATURE_NAMES, workload_features_for(history, 201, cutoff_ns)))

    assert features["days_since_last_appearance"] == pytest.approx(8.0)
    assert features["pitches_last_appearance"] == 25.0
    assert features["pitches_trailing_short"] == 0.0
    assert features["pitches_trailing_long"] == 0.0
    assert features["appearances_trailing_long"] == 0.0
    assert features["back_to_back"] == 0.0


def test_workload_features_for_detects_back_to_back_usage():
    appearances = pd.DataFrame(
        [
            {"game_pk": 1, "team": "DET", "pitcher_id": 300, "game_date": pd.Timestamp("2023-04-01"), "season": 2023, "is_starter": False},
            {"game_pk": 2, "team": "DET", "pitcher_id": 300, "game_date": pd.Timestamp("2023-04-02"), "season": 2023, "is_starter": False},
        ]
    )
    counts = pd.DataFrame([{"game_pk": 1, "pitcher_id": 300, "pitches": 10}, {"game_pk": 2, "pitcher_id": 300, "pitches": 12}])
    history = build_workload_history(appearances, counts)

    # As of 2023-04-03: pitcher 300 appeared on both 04-01 and 04-02 -- back-to-back.
    cutoff_ns = pd.Timestamp("2023-04-03").value
    features = dict(zip(WORKLOAD_FEATURE_NAMES, workload_features_for(history, 300, cutoff_ns)))
    assert features["back_to_back"] == 1.0


def test_closer_recency_features_for_unknown_pitcher_returns_sentinel():
    history = _history_fixture()
    save_recency, light_recency = closer_recency_features_for(history, 999999, pd.Timestamp("2023-04-02").value)
    assert save_recency == 365.0
    assert light_recency == 365.0


def test_closer_recency_features_for_matches_the_nearest_prior_save_and_light_dates():
    appearances = pd.DataFrame(
        [
            {"game_pk": 1, "team": "DET", "pitcher_id": 300, "game_date": pd.Timestamp("2023-04-01"), "season": 2023, "is_starter": False},
            {"game_pk": 2, "team": "DET", "pitcher_id": 300, "game_date": pd.Timestamp("2023-04-05"), "season": 2023, "is_starter": False},
        ]
    )
    counts = pd.DataFrame([{"game_pk": 1, "pitcher_id": 300, "pitches": 35}, {"game_pk": 2, "pitcher_id": 300, "pitches": 10}])
    entry_situations = pd.DataFrame(
        [
            {"game_pk": 1, "pitcher_id": 300, "entry_inning": 9, "is_save_situation": True},
            {"game_pk": 2, "pitcher_id": 300, "entry_inning": 7, "is_save_situation": False},
        ]
    )
    history = build_workload_history(appearances, counts, entry_situations)

    cutoff_ns = pd.Timestamp("2023-04-08").value  # 7 days after the light outing, 3 after the save
    save_recency, light_recency = closer_recency_features_for(history, 300, cutoff_ns)
    assert save_recency == pytest.approx(7.0)  # last save situation was game 1 (04-01)
    assert light_recency == pytest.approx(3.0)  # last light appearance was game 2 (04-05)


# ---------- team-level save-opportunity history (closer-only feature) ----------


def test_compute_team_save_opportunity_counts_sums_save_situations_per_team_game():
    appearances = pd.DataFrame(
        [
            {"game_pk": 1, "team": "DET", "pitcher_id": 300, "game_date": pd.Timestamp("2023-04-01"), "season": 2023, "is_starter": False},
            {"game_pk": 1, "team": "DET", "pitcher_id": 301, "game_date": pd.Timestamp("2023-04-01"), "season": 2023, "is_starter": False},
            {"game_pk": 2, "team": "DET", "pitcher_id": 300, "game_date": pd.Timestamp("2023-04-02"), "season": 2023, "is_starter": False},
        ]
    )
    # Game 1: both 300 and 301 face save situations (2 total). Game 2: none.
    entry_situations = pd.DataFrame(
        [
            {"game_pk": 1, "pitcher_id": 300, "is_save_situation": True},
            {"game_pk": 1, "pitcher_id": 301, "is_save_situation": True},
            {"game_pk": 2, "pitcher_id": 300, "is_save_situation": False},
        ]
    )
    counts = compute_team_save_opportunity_counts(appearances, entry_situations).set_index("game_pk")
    assert counts.loc[1, "save_situations"] == 2
    assert counts.loc[2, "save_situations"] == 0


def test_team_save_opportunities_trailing_for_sums_within_the_window():
    team_save_counts = pd.DataFrame(
        [
            {"team": "DET", "game_pk": 1, "game_date": pd.Timestamp("2023-04-01"), "save_situations": 1},
            {"team": "DET", "game_pk": 2, "game_date": pd.Timestamp("2023-04-03"), "save_situations": 2},
            {"team": "DET", "game_pk": 3, "game_date": pd.Timestamp("2023-03-01"), "save_situations": 5},  # outside the window
        ]
    )
    history = build_team_save_opportunity_history(team_save_counts)

    # As of 2023-04-08 with a 7-day trailing window: game 2 (04-03, within
    # 7 days) counts, game 1 (04-01, exactly 7 days back) counts, game 3
    # (03-01) is long outside the window.
    total = team_save_opportunities_trailing_for(history, "DET", pd.Timestamp("2023-04-08").value, trailing_days=7)
    assert total == pytest.approx(3.0)


def test_team_save_opportunities_trailing_for_unknown_team_returns_zero():
    history = build_team_save_opportunity_history(pd.DataFrame(columns=["team", "game_pk", "game_date", "save_situations"]))
    assert team_save_opportunities_trailing_for(history, "XXX", pd.Timestamp("2023-04-08").value) == 0.0


def test_attach_team_save_opportunity_feature_matches_direct_lookup():
    team_save_counts = pd.DataFrame(
        [{"team": "DET", "game_pk": 1, "game_date": pd.Timestamp("2023-04-01"), "save_situations": 1}]
    )
    history = build_team_save_opportunity_history(team_save_counts)
    examples = pd.DataFrame({"team": ["DET"], "game_date": [pd.Timestamp("2023-04-05")]})

    result = attach_team_save_opportunity_feature(examples, history)
    expected = team_save_opportunities_trailing_for(history, "DET", pd.Timestamp("2023-04-05").value)
    assert result.loc[0, TEAM_SAVE_OPPORTUNITY_FEATURE_NAME] == pytest.approx(expected)


def test_fit_logistic_regression_model_with_closer_only_features_has_the_right_coefficient_count():
    train = _synthetic_examples(n=500, seed=20)
    closer_train = train[train["role"] == "closer"]
    model = fit_logistic_regression_model(closer_train, CLOSER_ONLY_FEATURE_NAMES)
    assert model.coef_.shape[1] == len(CLOSER_ONLY_FEATURE_NAMES)


def test_closer_only_model_predictions_depend_on_team_save_opportunity_feature():
    train = _synthetic_examples(n=1000, seed=21)
    closer_train = train[train["role"] == "closer"]
    model = fit_logistic_regression_model(closer_train, CLOSER_ONLY_FEATURE_NAMES)

    from src.models.bullpen_availability import _closer_predict_proba

    row_a = closer_train.iloc[[0]].copy()
    row_b = row_a.copy()
    row_b[TEAM_SAVE_OPPORTUNITY_FEATURE_NAME] = row_a[TEAM_SAVE_OPPORTUNITY_FEATURE_NAME].iloc[0] + 5.0

    prob_a = _closer_predict_proba("closer_only_logistic_regression", model, row_a)[0]
    prob_b = _closer_predict_proba("closer_only_logistic_regression", model, row_b)[0]
    assert prob_a != pytest.approx(prob_b)


# ---------- build_query_examples ----------


def test_build_query_examples_labels_and_candidate_pool_for_game_2():
    history = _history_fixture()
    examples = build_query_examples(_pitcher_appearances_fixture(), history, window_days=14)

    game2 = examples[examples["game_pk"] == 2].set_index("pitcher_id")
    # Candidates: everyone who appeared for DET in the 14 days before game 2,
    # excluding game 2's own starter (101) -- that's 100 (yesterday's
    # starter, still eligible as a bullpen candidate), 200, and 201.
    assert set(game2.index) == {100, 200, 201}
    assert game2.loc[200, "label"] == 1  # actually relieved in game 2
    assert game2.loc[100, "label"] == 0  # candidate, but didn't appear
    assert game2.loc[201, "label"] == 0  # candidate, but didn't appear


def test_build_query_examples_excludes_only_that_games_own_starter():
    history = _history_fixture()
    examples = build_query_examples(_pitcher_appearances_fixture(), history, window_days=14)
    game1 = examples[examples["game_pk"] == 1]
    # Game 1's own starter (100) must never appear as a candidate for game 1 itself.
    assert 100 not in game1["pitcher_id"].tolist()


def test_build_query_examples_respects_the_candidate_window():
    """With a 3-day window instead of 14, pitcher 201 (last seen 8 days
    before game 2) should no longer be a candidate for game 2 at all."""
    history = _history_fixture()
    examples = build_query_examples(_pitcher_appearances_fixture(), history, window_days=3)
    game2 = examples[examples["game_pk"] == 2]
    assert 201 not in game2["pitcher_id"].tolist()
    assert 200 in game2["pitcher_id"].tolist()


# ---------- HeuristicAvailabilityModel ----------


def test_heuristic_more_rest_increases_availability():
    heuristic = HeuristicAvailabilityModel()
    base = {"days_since_last_appearance": 0.0, "pitches_last_appearance": 0.0, "back_to_back": 0.0}
    rested = {"days_since_last_appearance": 3.0, "pitches_last_appearance": 0.0, "back_to_back": 0.0}
    X = pd.DataFrame([base, rested])
    probs = heuristic.predict_proba(X)
    assert probs[1] > probs[0]


def test_heuristic_heavy_last_outing_decreases_availability():
    heuristic = HeuristicAvailabilityModel()
    light = {"days_since_last_appearance": 2.0, "pitches_last_appearance": 5.0, "back_to_back": 0.0}
    heavy = {"days_since_last_appearance": 2.0, "pitches_last_appearance": 45.0, "back_to_back": 0.0}
    X = pd.DataFrame([light, heavy])
    probs = heuristic.predict_proba(X)
    assert probs[1] < probs[0]


def test_heuristic_back_to_back_decreases_availability():
    heuristic = HeuristicAvailabilityModel()
    not_b2b = {"days_since_last_appearance": 1.0, "pitches_last_appearance": 10.0, "back_to_back": 0.0}
    b2b = {"days_since_last_appearance": 1.0, "pitches_last_appearance": 10.0, "back_to_back": 1.0}
    X = pd.DataFrame([not_b2b, b2b])
    probs = heuristic.predict_proba(X)
    assert probs[1] < probs[0]


# ---------- RoleAwareLogisticRegressionModel ----------


def _role_aware_examples(n=300, seed=0):
    """Synthetic examples where closers' true label depends heavily on
    save/light recency (features non-closers structurally can't use), and
    everyone's label also depends on the ordinary workload features."""
    rng = np.random.default_rng(seed)
    role = rng.choice(["closer", "middle_reliever", "long_reliever"], size=n, p=[0.15, 0.6, 0.25])
    days_since = rng.uniform(0, 4, n)
    pitches_last = rng.uniform(0, 50, n)
    save_recency = rng.uniform(0, 10, n)
    light_recency = rng.uniform(0, 10, n)

    is_closer = (role == "closer").astype(float)
    logit = (
        0.8 * days_since / 4
        - 0.6 * pitches_last / 50
        - 1.2 * is_closer * (save_recency / 10)
        - 0.8 * is_closer * (light_recency / 10)
    )
    prob = 1 / (1 + np.exp(-logit))
    label = (rng.uniform(size=n) < prob).astype(int)

    return pd.DataFrame(
        {
            "pitcher_id": np.arange(n),
            "role": role,
            "label": label,
            "days_since_last_appearance": days_since,
            "pitches_last_appearance": pitches_last,
            "pitches_trailing_short": pitches_last,
            "pitches_trailing_long": pitches_last,
            "appearances_trailing_long": np.ones(n),
            "back_to_back": np.zeros(n),
            CLOSER_RECENCY_FEATURE_NAMES[0]: save_recency,
            CLOSER_RECENCY_FEATURE_NAMES[1]: light_recency,
        }
    )


def test_role_aware_model_design_matrix_zeroes_recency_interactions_for_non_closers():
    model = RoleAwareLogisticRegressionModel()
    middle = pd.DataFrame(
        [{**dict.fromkeys(WORKLOAD_FEATURE_NAMES, 0.0), "role": "middle_reliever",
          CLOSER_RECENCY_FEATURE_NAMES[0]: 5.0, CLOSER_RECENCY_FEATURE_NAMES[1]: 3.0}]
    )
    X = model._design_matrix(middle)
    # Last two design-matrix columns are the closer interaction terms.
    assert X[0, -2] == 0.0
    assert X[0, -1] == 0.0


def test_role_aware_model_non_closer_predictions_are_invariant_to_recency_features():
    train = _role_aware_examples(seed=1)
    model = RoleAwareLogisticRegressionModel().fit(train)

    row_a = train[train["role"] == "middle_reliever"].iloc[[0]].copy()
    row_b = row_a.copy()
    row_b[CLOSER_RECENCY_FEATURE_NAMES[0]] = 999.0
    row_b[CLOSER_RECENCY_FEATURE_NAMES[1]] = 999.0

    assert model.predict_proba(row_a)[0] == pytest.approx(model.predict_proba(row_b)[0])


def test_role_aware_model_closer_predictions_do_depend_on_recency_features():
    train = _role_aware_examples(seed=2)
    model = RoleAwareLogisticRegressionModel().fit(train)

    row_a = train[train["role"] == "closer"].iloc[[0]].copy()
    row_b = row_a.copy()
    row_b[CLOSER_RECENCY_FEATURE_NAMES[0]] = 0.0
    row_b[CLOSER_RECENCY_FEATURE_NAMES[1]] = 0.0

    assert model.predict_proba(row_a)[0] != pytest.approx(model.predict_proba(row_b)[0])


# ---------- evaluate_predictions ----------


def test_evaluate_predictions_perfect_separation():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])
    metrics = evaluate_predictions(y_true, y_prob)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["mean_prob_when_appeared"] == pytest.approx(0.85)
    assert metrics["mean_prob_when_not_appeared"] == pytest.approx(0.15)
    assert metrics["hit_rate_when_appeared"] == pytest.approx(1.0)
    assert metrics["hit_rate_when_not_appeared"] == pytest.approx(1.0)


def test_evaluate_predictions_handles_all_one_class():
    y_true = np.array([1, 1, 1])
    y_prob = np.array([0.6, 0.7, 0.8])
    metrics = evaluate_predictions(y_true, y_prob)
    assert np.isnan(metrics["auc"])  # undefined with only one class present
    assert metrics["n_did_not_appear"] == 0


# ---------- compute_entry_innings / classify_reliever_roles ----------


def _role_fixture_appearances_and_counts(n_appearances=12):
    """Three relievers with clearly distinct usage patterns: 900 is a
    textbook closer (always the 9th, short outings), 901 a middle reliever
    (mid-game, moderate pitches), 902 a long reliever (early entry, heavy
    pitch counts). Pitcher 903 only has 3 appearances -- below the default
    min_appearances threshold, so it should end up unclassified."""
    appearances = []
    counts = []
    profiles = {
        900: {"inning": 9, "pitches": 15},
        901: {"inning": 7, "pitches": 20},
        902: {"inning": 2, "pitches": 45},
        903: {"inning": 9, "pitches": 12},
    }
    for pitcher_id, profile in profiles.items():
        n = 3 if pitcher_id == 903 else n_appearances
        for i in range(n):
            game_pk = pitcher_id * 100 + i
            appearances.append(
                {"game_pk": game_pk, "team": "DET", "pitcher_id": pitcher_id,
                 "game_date": pd.Timestamp("2023-04-01") + pd.Timedelta(days=i), "season": 2023, "is_starter": False}
            )
            counts.append({"game_pk": game_pk, "pitcher_id": pitcher_id, "pitches": profile["pitches"]})
    entry_innings = pd.DataFrame(
        [{"game_pk": a["game_pk"], "pitcher_id": a["pitcher_id"], "entry_inning": profiles[a["pitcher_id"]]["inning"]} for a in appearances]
    )
    return pd.DataFrame(appearances), entry_innings, pd.DataFrame(counts)


def test_classify_reliever_roles_labels_closer_middle_and_long():
    appearances, entry_innings, counts = _role_fixture_appearances_and_counts()
    roles = classify_reliever_roles(appearances, entry_innings, counts, min_appearances=10)

    assert roles[900] == "closer"
    assert roles[901] == "middle_reliever"
    assert roles[902] == "long_reliever"


def test_classify_reliever_roles_excludes_pitchers_below_min_appearances():
    appearances, entry_innings, counts = _role_fixture_appearances_and_counts()
    roles = classify_reliever_roles(appearances, entry_innings, counts, min_appearances=10)
    assert 903 not in roles.index


def test_predictor_normalizes_roles_series_to_a_plain_dict_matching_original_pandas_lookup_for_every_pitcher_id():
    # BullpenAvailabilityPredictor.__post_init__ replaces a pd.Series roles
    # table with a plain dict[int, str] (see LeagueRatesIndex/
    # BaserunningModel._pooled_index for the same pre-indexed-dict pattern
    # applied earlier against park factors and base-running). This is an
    # oracle test: the dict-backed _role_for must return exactly what a
    # direct pandas Series.get(..., "unclassified") lookup would have,
    # across every pitcher id the Series covers plus ids it doesn't.
    appearances, entry_innings, counts = _role_fixture_appearances_and_counts()
    roles_series = classify_reliever_roles(appearances, entry_innings, counts, min_appearances=10)
    assert isinstance(roles_series, pd.Series)  # sanity: the oracle input really is pandas-backed

    predictor = BullpenAvailabilityPredictor(
        kind="heuristic", model=HeuristicAvailabilityModel(), config=BullpenAvailabilityConfig(), roles=roles_series
    )
    assert isinstance(predictor.roles, dict)

    queried_ids = list(roles_series.index) + [999999, -1]  # + two ids absent from the Series
    for pitcher_id in queried_ids:
        expected = roles_series.get(pitcher_id, "unclassified")
        assert predictor._role_for(pitcher_id) == expected
        assert predictor.roles.get(pitcher_id, "unclassified") == expected


# ---------- attach_roles / attach_closer_recency_features ----------


def test_attach_roles_fills_unclassified_for_unknown_pitchers():
    examples = pd.DataFrame({"pitcher_id": [900, 999999]})
    roles = pd.Series({900: "closer"})
    result = attach_roles(examples, roles)
    assert result.loc[0, "role"] == "closer"
    assert result.loc[1, "role"] == "unclassified"


def test_attach_closer_recency_features_matches_direct_lookup():
    appearances = pd.DataFrame(
        [{"game_pk": 1, "team": "DET", "pitcher_id": 300, "game_date": pd.Timestamp("2023-04-01"), "season": 2023, "is_starter": False}]
    )
    counts = pd.DataFrame([{"game_pk": 1, "pitcher_id": 300, "pitches": 10}])
    entry_situations = pd.DataFrame([{"game_pk": 1, "pitcher_id": 300, "entry_inning": 9, "is_save_situation": True}])
    history = build_workload_history(appearances, counts, entry_situations)

    examples = pd.DataFrame({"pitcher_id": [300], "game_date": [pd.Timestamp("2023-04-05")]})
    result = attach_closer_recency_features(examples, history)

    expected = closer_recency_features_for(history, 300, pd.Timestamp("2023-04-05").value)
    assert result.loc[0, CLOSER_RECENCY_FEATURE_NAMES[0]] == pytest.approx(expected[0])
    assert result.loc[0, CLOSER_RECENCY_FEATURE_NAMES[1]] == pytest.approx(expected[1])


# ---------- compute_calibration_table / calibration_by_role ----------


def test_compute_calibration_table_bins_match_hand_computed_values():
    # 10 samples, 2 bins -- bin 0 (low predicted prob) always label 0, bin 1
    # (high predicted prob) always label 1, so this is fully separable and
    # each bin's numbers are easy to hand-check.
    y_prob = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.7, 0.75, 0.8, 0.85, 0.9])
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    table = compute_calibration_table(y_true, y_prob, n_bins=2)

    assert len(table) == 2
    assert table.loc[0, "n"] == 5
    assert table.loc[0, "mean_predicted"] == pytest.approx(0.2)
    assert table.loc[0, "observed_rate"] == pytest.approx(0.0)
    assert table.loc[1, "mean_predicted"] == pytest.approx(0.8)
    assert table.loc[1, "observed_rate"] == pytest.approx(1.0)


def test_compute_calibration_table_gap_is_predicted_minus_observed():
    y_prob = np.array([0.9, 0.9, 0.9, 0.9])  # confidently "available"...
    y_true = np.array([0, 0, 0, 0])  # ...but nobody actually appeared: badly miscalibrated
    table = compute_calibration_table(y_true, y_prob, n_bins=1)
    assert table.loc[0, "calibration_gap"] == pytest.approx(0.9)


def test_mean_abs_calibration_gap_averages_absolute_gaps():
    table = pd.DataFrame({"calibration_gap": [0.1, -0.3, 0.2]})
    assert mean_abs_calibration_gap(table) == pytest.approx((0.1 + 0.3 + 0.2) / 3)


def test_calibration_by_role_includes_overall_and_skips_undersized_roles():
    n = 40
    examples = pd.DataFrame(
        {
            "pitcher_id": [900] * 20 + [901] * 15 + [902] * 5,  # 902 has too few for n_bins=10
            "label": ([1, 0] * 10) + ([1, 0] * 7 + [1]) + [1, 0, 1, 0, 1],
        }
    )
    y_prob = np.linspace(0.05, 0.95, n)
    roles = pd.Series({900: "closer", 901: "middle_reliever", 902: "long_reliever"})

    tables = calibration_by_role(examples, y_prob, roles, n_bins=10)

    assert "overall" in tables
    assert "closer" in tables
    assert "middle_reliever" in tables
    assert "long_reliever" not in tables  # only 5 examples, below n_bins=10


# ---------- plot_calibration_reliability ----------


def test_plot_calibration_reliability_writes_a_file(tmp_path):
    table = compute_calibration_table(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), n_bins=2
    )
    output_path = tmp_path / "nested" / "calibration.png"
    plot_calibration_reliability({"overall": table}, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# ---------- select_availability_model ----------


def _synthetic_examples(n=200, seed=0, include_closers=True):
    rng = np.random.default_rng(seed)
    role = (
        rng.choice(["closer", "middle_reliever", "long_reliever"], size=n, p=[0.15, 0.6, 0.25])
        if include_closers
        else np.full(n, "middle_reliever")
    )
    days_since = rng.uniform(0, 4, n)
    pitches_last = rng.uniform(0, 50, n)
    back_to_back = rng.integers(0, 2, n).astype(float)
    save_recency = rng.uniform(0, 10, n)
    light_recency = rng.uniform(0, 10, n)
    team_recent_saves = rng.uniform(0, 5, n)
    is_closer = (role == "closer").astype(float)

    logit = (
        1.2 * days_since / 4
        - 1.0 * pitches_last / 50
        - 1.5 * back_to_back
        - 1.0 * is_closer * (save_recency / 10)
        + 0.8 * is_closer * (team_recent_saves / 5)
    )
    prob = 1 / (1 + np.exp(-logit))
    label = (rng.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "pitcher_id": np.arange(n),
            "team": "DET",
            "game_pk": np.arange(n),
            "game_date": pd.Timestamp("2023-04-01"),
            "season": 2023,
            "label": label,
            "role": role,
            "days_since_last_appearance": days_since,
            "pitches_last_appearance": pitches_last,
            "pitches_trailing_short": pitches_last,
            "pitches_trailing_long": pitches_last,
            "appearances_trailing_long": np.ones(n),
            "back_to_back": back_to_back,
            CLOSER_RECENCY_FEATURE_NAMES[0]: save_recency,
            CLOSER_RECENCY_FEATURE_NAMES[1]: light_recency,
            TEAM_SAVE_OPPORTUNITY_FEATURE_NAME: team_recent_saves,
        }
    )
    return df


def _roles_from_examples(examples: pd.DataFrame) -> pd.Series:
    return examples.drop_duplicates("pitcher_id").set_index("pitcher_id")["role"]


def test_select_availability_model_picks_the_higher_auc_general_candidate():
    train = _synthetic_examples(seed=1)
    val = _synthetic_examples(seed=2)
    roles = _roles_from_examples(pd.concat([train, val]))
    result = select_availability_model(train, val, roles, BullpenAvailabilityConfig())

    assert isinstance(result, ModelSelectionResult)
    assert result.general_kind in {"heuristic", "logistic_regression"}
    if result.logistic_metrics["auc"] > result.heuristic_metrics["auc"]:
        assert result.general_kind == "logistic_regression"
        assert isinstance(result.predictor.model, LogisticRegression)
    else:
        assert result.general_kind == "heuristic"
        assert isinstance(result.predictor.model, HeuristicAvailabilityModel)


def test_select_availability_model_both_general_metrics_are_reasonable_on_learnable_data():
    train = _synthetic_examples(seed=3)
    val = _synthetic_examples(seed=4)
    roles = _roles_from_examples(pd.concat([train, val]))
    result = select_availability_model(train, val, roles, BullpenAvailabilityConfig())
    assert result.heuristic_metrics["auc"] > 0.55
    assert result.logistic_metrics["auc"] > 0.55


def test_select_availability_model_picks_a_closer_path_and_its_the_better_calibrated_one():
    train = _synthetic_examples(n=1000, seed=5)
    val = _synthetic_examples(n=1000, seed=6)
    roles = _roles_from_examples(pd.concat([train, val]))
    result = select_availability_model(train, val, roles, BullpenAvailabilityConfig())

    assert result.closer_kind in {"closer_only_logistic_regression", "role_aware_logistic_regression"}
    closer_only_gap = mean_abs_calibration_gap(result.closer_only_calibration)
    role_aware_gap = mean_abs_calibration_gap(result.role_aware_closer_calibration)
    if result.closer_kind == "closer_only_logistic_regression":
        assert closer_only_gap <= role_aware_gap
    else:
        assert role_aware_gap < closer_only_gap

    assert result.predictor.closer_model is not None
    assert result.predictor.closer_kind == result.closer_kind


def test_select_availability_model_skips_closer_path_with_no_closers():
    train = _synthetic_examples(n=200, seed=7, include_closers=False)
    val = _synthetic_examples(n=200, seed=8, include_closers=False)
    roles = _roles_from_examples(pd.concat([train, val]))
    result = select_availability_model(train, val, roles, BullpenAvailabilityConfig())

    assert result.closer_kind == "none"
    assert result.predictor.closer_model is None
    assert result.predictor.closer_kind is None


# ---------- BullpenAvailabilityPredictor + persistence ----------


def test_predictor_predict_proba_matches_batch_prediction_general_model():
    train = _synthetic_examples(n=1000, seed=9)
    val = _synthetic_examples(n=1000, seed=10)
    roles = _roles_from_examples(pd.concat([train, val]))
    result = select_availability_model(train, val, roles, BullpenAvailabilityConfig())

    history = _history_fixture()
    # Pitcher 200 in the module's own workload-history fixture is not
    # classified in `roles`, so this exercises the general (non-closer) path.
    single = result.predictor.predict_proba(history, pitcher_id=200, as_of_date="2023-04-02")

    row = pd.DataFrame([workload_features_for(history, 200, pd.Timestamp("2023-04-02").value)], columns=WORKLOAD_FEATURE_NAMES)
    row["role"] = "unclassified"
    batch = result.predictor.predict_proba_batch(row)
    assert single == pytest.approx(batch[0])


def test_predictor_dispatches_closer_examples_to_the_closer_model():
    train = _synthetic_examples(n=1000, seed=11)
    val = _synthetic_examples(n=1000, seed=12)
    roles = _roles_from_examples(pd.concat([train, val]))
    result = select_availability_model(train, val, roles, BullpenAvailabilityConfig())

    closer_rows = val[val["role"] == "closer"]
    assert len(closer_rows) > 0
    batch_probs = result.predictor.predict_proba_batch(closer_rows)

    # Directly recomputed via the closer-only dispatch path should match exactly.
    from src.models.bullpen_availability import _closer_predict_proba

    expected = _closer_predict_proba(result.closer_kind, result.predictor.closer_model, closer_rows)
    assert np.allclose(batch_probs, expected)


def test_save_and_load_predictor_round_trips(tmp_path):
    train = _synthetic_examples(n=1000, seed=13)
    val = _synthetic_examples(n=1000, seed=14)
    roles = _roles_from_examples(pd.concat([train, val]))
    result = select_availability_model(train, val, roles, BullpenAvailabilityConfig(window_days=10))

    path = tmp_path / "predictor.pkl"
    save_predictor(result.predictor, path)
    loaded = load_predictor(path)

    assert loaded.kind == result.predictor.kind
    assert loaded.closer_kind == result.predictor.closer_kind
    assert loaded.config.window_days == 10

    history = _history_fixture()
    assert loaded.predict_proba(history, 200, "2023-04-02") == pytest.approx(
        result.predictor.predict_proba(history, 200, "2023-04-02")
    )


# ---------- main() end-to-end ----------


def _raw_row(pitcher, batter, game_date, at_bat_number, pitch_number, inning_topbot, season,
             home_team="DET", away_team="CLE", inning=1, home_score=0, away_score=0):
    return {
        "pitcher": pitcher,
        "batter": batter,
        "game_date": game_date,
        "game_pk": season,  # overwritten per-game in _write_fixture
        "game_year": season,
        "game_type": "R",
        "home_team": home_team,
        "away_team": away_team,
        "inning_topbot": inning_topbot,
        "inning": inning,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "pitch_type": "FF",
        "release_speed": 90.0,
        "release_spin_rate": 2200,
        "spin_rate_deprecated": None,
        "plate_x": 0.1,
        "plate_z": 2.2,
        "balls": 0,
        "strikes": 0,
        "outs_when_up": 0,
        "on_1b": None,
        "on_2b": None,
        "on_3b": None,
        "home_score": home_score,
        "away_score": away_score,
        "post_home_score": 3,
        "post_away_score": 1,
        "n_thruorder_pitcher": 1,
        "stand": "R",
        "p_throws": "L",
        "events": "field_out",
        "description": "hit_into_play",
    }


def _write_fixture(raw_dir, pitches_dir):
    """Three DET-home-vs-CLE-away games per season (2015-2023), a few days
    apart -- close enough together that later games in the same season have
    real bullpen candidates from earlier ones. DET starter 100 pitches the
    top half every game; reliever 200 (a long-reliever profile: early
    entry, heavy pitch count) relieves in games 1 and 3 but not game 2, so
    build_query_examples sees both a positive and a negative label for the
    same candidate. DET closer 300 enters the 9th in a save situation
    (home leading 3-1, inning_topbot="Top" so DET is fielding) with a light
    pitch count, also in games 1 and 3 -- enough appearances across 9
    seasons to classify as "closer" and exercise the closer-specific path
    end-to-end. CLE starter 900 pitches the bottom half so
    _build_season_game_tables records a real away starter too -- otherwise
    every game gets dropped for a missing away_starter_id.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    seasons = list(range(TRAIN_SEASON_RANGE[0], TRAIN_SEASON_RANGE[1] + 1)) + list(VAL_SEASONS)
    for season in seasons:
        season_rows = []
        for game_idx, (day, reliever_appears) in enumerate([(1, True), (4, False), (7, True)]):
            date = f"{season}-04-{day:02d}"
            game_rows = [
                _raw_row(100, 101 + i, date, i + 1, 1, "Top", season) for i in range(4)
            ]
            if reliever_appears:
                game_rows += [_raw_row(200, 105 + i, date, 5 + i, 1, "Top", season) for i in range(2)]
                game_rows += [
                    _raw_row(300, 108 + i, date, 8 + i, 1, "Top", season, inning=9, home_score=3, away_score=1)
                    for i in range(2)
                ]
            game_rows += [_raw_row(900, 110 + i, date, 20 + i, 1, "Bot", season) for i in range(4)]
            for row in game_rows:
                row["game_pk"] = season * 10 + game_idx
            season_rows.extend(game_rows)
        pd.DataFrame(season_rows).to_parquet(raw_dir / f"statcast_{season}.parquet")
        all_rows.extend(season_rows)

    raw_all = pd.DataFrame(all_rows)
    pitches = build_season_pitches_from_frame(build_pitch_frame_from_raw(raw_all))
    write_partitioned(pitches, pitches_dir)
    return pitches


def test_main_runs_end_to_end_and_saves_a_checkpoint(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    _write_fixture(raw_dir, pitches_dir)

    games_dir = tmp_path / "games"
    pitcher_appearances_dir = tmp_path / "pitcher_appearances"
    batter_appearances_dir = tmp_path / "batter_appearances"
    checkpoint_path = tmp_path / "bullpen_availability.pkl"
    calibration_plot_path = tmp_path / "calibration.png"

    bullpen_main(
        [
            "--pitches-dir", str(pitches_dir),
            "--raw-dir", str(raw_dir),
            "--games-dir", str(games_dir),
            "--pitcher-appearances-dir", str(pitcher_appearances_dir),
            "--batter-appearances-dir", str(batter_appearances_dir),
            "--checkpoint", str(checkpoint_path),
            "--calibration-plot", str(calibration_plot_path),
            "--calibration-bins", "2",
        ]
    )

    assert checkpoint_path.exists()
    assert calibration_plot_path.exists()
    predictor = load_predictor(checkpoint_path)
    assert predictor.kind in {"heuristic", "logistic_regression"}
    # Pitcher 300 (9 seasons x 2 appearances/season = 18 relief appearances,
    # all in save situations) should classify as a closer and trigger the
    # closer-specific path.
    assert predictor.roles is not None
    assert predictor.roles.get(300) == "closer"
    assert predictor.closer_kind in {"closer_only_logistic_regression", "role_aware_logistic_regression"}
    assert predictor.closer_model is not None
