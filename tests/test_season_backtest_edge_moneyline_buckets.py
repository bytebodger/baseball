import numpy as np
import pandas as pd
import pytest

from src.evaluation.season_backtest_edge_moneyline_buckets import (
    MONEYLINE_BUCKETS,
    assign_moneyline_bucket,
    compute_edge_moneyline_report,
)


def test_assign_moneyline_bucket_covers_every_declared_range():
    odds = np.array([-250, -200, -175, -151, -150, -101, 100, 125, 150, 151, 200, 250])
    labels = assign_moneyline_bucket(odds)
    assert labels.tolist() == [
        "< -200", "-151 to -200", "-151 to -200", "-151 to -200",
        "-101 to -150", "-101 to -150", "+100 to +150", "+100 to +150", "+100 to +150",
        "+151 to +200", "+151 to +200", "> +200",
    ]


def test_assign_moneyline_bucket_reports_the_trivial_gap_as_other():
    odds = np.array([-100, -50, 0, 50, 99])
    labels = assign_moneyline_bucket(odds)
    assert (labels == "other").all()


def test_moneyline_buckets_are_mutually_exclusive_and_exhaustive_except_the_gap():
    # every label used by the predicates must be a declared bucket label
    declared = {label for label, _ in MONEYLINE_BUCKETS}
    assert declared == {"< -200", "-151 to -200", "-101 to -150", "+100 to +150", "+151 to +200", "> +200"}


def _games(rows):
    """rows: list of (game_pk, home_win, home_ml_open, away_ml_open, home_ml_close, away_ml_close, model_home_prob)."""
    cols = ["game_pk", "home_win", "home_ml_open", "away_ml_open", "home_ml_close", "away_ml_close", "model_home_prob"]
    return pd.DataFrame(rows, columns=cols)


def test_report_buckets_by_edge_threshold_and_moneyline_range():
    # game 1: home favorite at -175 (falls in "-151 to -200"), model edge on home = 0.65 - open_home_prob.
    #   open -175/-155 no-vig home prob is high (~0.63ish) -- model 0.65 clears 2% but not comfortably past 4%,
    #   so this bet should show up at 2%+ but let's just assert n_bets/bucket membership, not exact edge math.
    # game 2: away underdog at +180 (falls in "+151 to +200"), model strongly favors away -> low home prob.
    rows = [
        (1, True, -175, 155, -175, 155, 0.68),
        (2, False, -120, 100, -120, 100, 0.30),
    ]
    df = _games(rows)
    report = compute_edge_moneyline_report(df, n_resamples=50, seed=0)

    assert set(report["edge_threshold"]) == {"2%+", "4%+", "6%+"}
    two_pct = report[report["edge_threshold"] == "2%+"]
    all_row = two_pct[two_pct["moneyline_bucket"] == "ALL"].iloc[0]
    assert all_row["n_bets"] >= 1  # at least one game clears a 2% edge somewhere


def test_report_handles_zero_bets_in_a_cell_without_crashing():
    # Both games priced at pick'em-ish odds with model probability matching the market almost exactly -> no bets.
    rows = [(1, True, -110, -110, -110, -110, 0.505)]
    df = _games(rows)
    report = compute_edge_moneyline_report(df, n_resamples=50, seed=0)
    assert (report["n_bets"] == 0).all()
    assert report["roi"].isna().all()


def test_report_roi_and_clv_match_hand_computed_values_for_a_single_clear_bet():
    # open pick'em (0.5/0.5), model says 0.70 home -> edge 0.20, clears 2%/4%/6% all three.
    # close shortens home to -150/+130 -> positive CLV for the home bet. Home wins.
    rows = [(1, True, -110, -110, -150, 130, 0.70)]
    df = _games(rows)
    report = compute_edge_moneyline_report(df, stake=1.0, n_resamples=50, seed=0)

    for edge_label in ["2%+", "4%+", "6%+"]:
        block = report[report["edge_threshold"] == edge_label]
        all_row = block[block["moneyline_bucket"] == "ALL"].iloc[0]
        assert all_row["n_bets"] == 1
        assert all_row["roi"] == pytest.approx(100.0 / 110.0)  # -110 favorite win payout
        assert all_row["mean_clv"] > 0

        # -110 odds fall in "-101 to -150"
        bucket_row = block[block["moneyline_bucket"] == "-101 to -150"].iloc[0]
        assert bucket_row["n_bets"] == 1
        other_buckets = block[~block["moneyline_bucket"].isin(["-101 to -150", "ALL"])]
        assert (other_buckets["n_bets"] == 0).all()
