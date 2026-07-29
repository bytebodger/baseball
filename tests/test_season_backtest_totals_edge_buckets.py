import numpy as np
import pandas as pd
import pytest

from src.evaluation.season_backtest_totals_edge_buckets import (
    EDGE_BUCKETS,
    assign_edge_bucket,
    compute_totals_edge_report,
    compute_totals_edge_side_report,
    simulate_totals_betting,
)


def _games(rows):
    """rows: list of dicts with total_open, over_open_odds, under_open_odds,
    total_close, over_close_odds, under_close_odds, home_score, away_score."""
    return pd.DataFrame(rows)


def test_assign_edge_bucket_covers_declared_ranges_and_reports_the_gap():
    edges = np.array([0.0, 0.5, 1.0, 1.2, 1.5, 1.75, 2.0, 2.1, 5.0])
    labels = assign_edge_bucket(edges)
    assert labels.tolist() == [
        "up to 1 run", "up to 1 run", "up to 1 run", "other",
        "1.5-2 runs", "1.5-2 runs", "1.5-2 runs", "more than 2 runs", "more than 2 runs",
    ]


def test_simulate_totals_betting_bets_over_when_model_predicts_higher_total():
    games = _games(
        [{"total_open": 8.5, "over_open_odds": -110, "under_open_odds": -110,
          "total_close": 8.5, "over_close_odds": -110, "under_close_odds": -110,
          "home_score": 5, "away_score": 4}]  # actual total = 9, over wins
    )
    bets = simulate_totals_betting(games, sim_mean_total_runs=[9.5], stake=1.0)
    assert bets.loc[0, "side"] == "over"
    assert bets.loc[0, "placed"]
    assert bets.loc[0, "edge"] == pytest.approx(1.0)
    assert not bets.loc[0, "is_push"]
    assert bets.loc[0, "profit"] == pytest.approx(100.0 / 110.0)  # -110 winner payout


def test_simulate_totals_betting_bets_under_when_model_predicts_lower_total():
    games = _games(
        [{"total_open": 9.5, "over_open_odds": -110, "under_open_odds": -110,
          "total_close": 9.5, "over_close_odds": -110, "under_close_odds": -110,
          "home_score": 3, "away_score": 4}]  # actual total = 7, under wins
    )
    bets = simulate_totals_betting(games, sim_mean_total_runs=[7.0], stake=1.0)
    assert bets.loc[0, "side"] == "under"
    assert bets.loc[0, "edge"] == pytest.approx(-2.5)
    assert bets.loc[0, "profit"] == pytest.approx(100.0 / 110.0)


def test_simulate_totals_betting_push_when_actual_total_equals_open_line():
    games = _games(
        [{"total_open": 9.0, "over_open_odds": -110, "under_open_odds": -110,
          "total_close": 9.0, "over_close_odds": -110, "under_close_odds": -110,
          "home_score": 5, "away_score": 4}]  # actual total = 9 == total_open -> push
    )
    bets = simulate_totals_betting(games, sim_mean_total_runs=[10.0], stake=1.0)
    assert bets.loc[0, "placed"]
    assert bets.loc[0, "is_push"]
    assert bets.loc[0, "profit"] == pytest.approx(0.0)


def test_simulate_totals_betting_no_bet_when_edge_is_exactly_zero():
    games = _games(
        [{"total_open": 8.5, "over_open_odds": -110, "under_open_odds": -110,
          "total_close": 8.5, "over_close_odds": -110, "under_close_odds": -110,
          "home_score": 5, "away_score": 4}]
    )
    bets = simulate_totals_betting(games, sim_mean_total_runs=[8.5], stake=1.0)
    assert bets.loc[0, "side"] is None
    assert not bets.loc[0, "placed"]
    assert np.isnan(bets.loc[0, "profit"])


def test_simulate_totals_betting_clv_positive_when_price_moves_toward_bet_side():
    # Bet over at -110; by close, over is favored more heavily (-140) -- price moved in the over bettor's favor.
    games = _games(
        [{"total_open": 8.5, "over_open_odds": -110, "under_open_odds": -110,
          "total_close": 8.5, "over_close_odds": -140, "under_close_odds": 120,
          "home_score": 5, "away_score": 5}]  # actual total = 10, over wins
    )
    bets = simulate_totals_betting(games, sim_mean_total_runs=[9.5], stake=1.0)
    assert bets.loc[0, "side"] == "over"
    assert bets.loc[0, "clv"] > 0


def test_missing_required_column_raises():
    games = pd.DataFrame({"total_open": [8.5]})
    with pytest.raises(ValueError):
        simulate_totals_betting(games, sim_mean_total_runs=[9.0])


def test_compute_totals_edge_report_buckets_and_excludes_pushes_from_roi():
    rows = [
        # small edge (0.3 runs), over wins
        {"total_open": 8.0, "over_open_odds": -110, "under_open_odds": -110,
         "total_close": 8.0, "over_close_odds": -110, "under_close_odds": -110,
         "home_score": 5, "away_score": 4, "sim_mean_total_runs": 8.3},
        # large edge (3.0 runs), over wins
        {"total_open": 7.0, "over_open_odds": -110, "under_open_odds": -110,
         "total_close": 7.0, "over_close_odds": -110, "under_close_odds": -110,
         "home_score": 6, "away_score": 5, "sim_mean_total_runs": 10.0},
        # push at a large edge magnitude -- must not contaminate ROI
        {"total_open": 9.0, "over_open_odds": -110, "under_open_odds": -110,
         "total_close": 9.0, "over_close_odds": -110, "under_close_odds": -110,
         "home_score": 5, "away_score": 4, "sim_mean_total_runs": 12.0},
    ]
    df = pd.DataFrame(rows)
    report = compute_totals_edge_report(df, n_resamples=50, seed=0)

    small = report[report["edge_bucket"] == "up to 1 run"].iloc[0]
    assert small["n_bets"] == 1
    assert small["roi"] == pytest.approx(100.0 / 110.0)

    large = report[report["edge_bucket"] == "more than 2 runs"].iloc[0]
    assert large["n_bets"] == 1  # the push is excluded from n_bets even though its edge is also >2
    assert large["n_pushes"] == 1
    assert large["roi"] == pytest.approx(100.0 / 110.0)

    all_row = report[report["edge_bucket"] == "ALL"].iloc[0]
    assert all_row["n_bets"] == 2
    assert all_row["n_pushes"] == 1


def test_compute_totals_edge_report_reports_the_gap_bucket_explicitly_not_just_folded_into_all():
    # edge = 1.2 runs -- lands in the undeclared 1.0-1.5 gap, must show up as its own "other" row,
    # not just silently counted inside "ALL" with no visible row of its own.
    rows = [
        {"total_open": 8.0, "over_open_odds": -110, "under_open_odds": -110,
         "total_close": 8.0, "over_close_odds": -110, "under_close_odds": -110,
         "home_score": 5, "away_score": 4, "sim_mean_total_runs": 9.2},
    ]
    df = pd.DataFrame(rows)
    report = compute_totals_edge_report(df, n_resamples=50, seed=0)
    assert "other" in report["edge_bucket"].tolist()
    other_row = report[report["edge_bucket"] == "other"].iloc[0]
    assert other_row["n_bets"] == 1
    for label, _ in EDGE_BUCKETS:
        assert report[report["edge_bucket"] == label].iloc[0]["n_bets"] == 0


# ---------- bias_correction ----------


def test_bias_correction_is_subtracted_from_prediction_before_computing_edge_and_side():
    games = _games(
        [{"total_open": 8.5, "over_open_odds": -110, "under_open_odds": -110,
          "total_close": 8.5, "over_close_odds": -110, "under_close_odds": -110,
          "home_score": 4, "away_score": 4}]  # actual total = 8, under wins
    )
    # raw prediction 9.23 -> normally an OVER bet (edge +0.73). With bias_correction=0.33,
    # the adjusted prediction is 8.90 -> still OVER, but a smaller edge (+0.40).
    bets = simulate_totals_betting(games, sim_mean_total_runs=[9.23], stake=1.0, bias_correction=0.33)
    assert bets.loc[0, "side"] == "over"
    assert bets.loc[0, "edge"] == pytest.approx(0.40, abs=1e-6)


def test_bias_correction_can_flip_the_bet_side():
    games = _games(
        [{"total_open": 9.0, "over_open_odds": -110, "under_open_odds": -110,
          "total_close": 9.0, "over_close_odds": -110, "under_close_odds": -110,
          "home_score": 5, "away_score": 5}]
    )
    # raw prediction 9.2 -> OVER (edge +0.2) without correction...
    uncorrected = simulate_totals_betting(games, sim_mean_total_runs=[9.2], stake=1.0, bias_correction=0.0)
    assert uncorrected.loc[0, "side"] == "over"
    # ...but 9.2 - 0.33 = 8.87 -> UNDER (edge -0.13) with the correction applied.
    corrected = simulate_totals_betting(games, sim_mean_total_runs=[9.2], stake=1.0, bias_correction=0.33)
    assert corrected.loc[0, "side"] == "under"
    assert corrected.loc[0, "edge"] == pytest.approx(-0.13, abs=1e-6)


def test_zero_bias_correction_reproduces_original_behavior():
    games = _games(
        [{"total_open": 8.5, "over_open_odds": -110, "under_open_odds": -110,
          "total_close": 8.5, "over_close_odds": -110, "under_close_odds": -110,
          "home_score": 5, "away_score": 4}]
    )
    default = simulate_totals_betting(games, sim_mean_total_runs=[9.5], stake=1.0)
    explicit_zero = simulate_totals_betting(games, sim_mean_total_runs=[9.5], stake=1.0, bias_correction=0.0)
    pd.testing.assert_frame_equal(default, explicit_zero)


# ---------- compute_totals_edge_side_report ----------


def test_edge_side_report_splits_over_and_under_within_each_edge_bucket():
    rows = [
        # small edge, over
        {"total_open": 8.0, "over_open_odds": -110, "under_open_odds": -110,
         "total_close": 8.0, "over_close_odds": -110, "under_close_odds": -110,
         "home_score": 5, "away_score": 4, "sim_mean_total_runs": 8.3},
        # small edge, under
        {"total_open": 9.0, "over_open_odds": -110, "under_open_odds": -110,
         "total_close": 9.0, "over_close_odds": -110, "under_close_odds": -110,
         "home_score": 4, "away_score": 3, "sim_mean_total_runs": 8.7},
    ]
    df = pd.DataFrame(rows)
    report = compute_totals_edge_side_report(df, n_resamples=50, seed=0)

    assert set(report["side"].unique()) == {"over", "under"}
    small_over = report[(report["edge_bucket"] == "up to 1 run") & (report["side"] == "over")].iloc[0]
    small_under = report[(report["edge_bucket"] == "up to 1 run") & (report["side"] == "under")].iloc[0]
    assert small_over["n_bets"] == 1
    assert small_under["n_bets"] == 1

    all_over = report[(report["edge_bucket"] == "ALL") & (report["side"] == "over")].iloc[0]
    all_under = report[(report["edge_bucket"] == "ALL") & (report["side"] == "under")].iloc[0]
    assert all_over["n_bets"] == 1
    assert all_under["n_bets"] == 1


def test_edge_side_report_accepts_bias_correction():
    games = _games(
        [{"total_open": 9.0, "over_open_odds": -110, "under_open_odds": -110,
          "total_close": 9.0, "over_close_odds": -110, "under_close_odds": -110,
          "home_score": 5, "away_score": 5}]
    )
    df = games.assign(sim_mean_total_runs=9.2)
    report = compute_totals_edge_side_report(df, n_resamples=50, seed=0, bias_correction=0.33)
    all_under = report[(report["edge_bucket"] == "ALL") & (report["side"] == "under")].iloc[0]
    all_over = report[(report["edge_bucket"] == "ALL") & (report["side"] == "over")].iloc[0]
    assert all_under["n_bets"] == 1  # bias correction flips 9.2 -> 8.87, below the 9.0 line
    assert all_over["n_bets"] == 0
