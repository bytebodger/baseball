import numpy as np
import pandas as pd
import pytest

from src.evaluation.season_backtest_metrics import (
    TIER_LABELS,
    assign_tier,
    compute_backtest_metrics,
    compute_pooled_and_tiered_metrics,
)


def test_assign_tier_boundaries_are_exactly_lt7_7to9_inclusive_gt9():
    values = [6.99, 7.0, 8.0, 9.0, 9.01, 0.0, 20.0]
    tiers = assign_tier(values)
    assert tiers.tolist() == ["<7", "7-9", "7-9", "7-9", ">9", "<7", ">9"]


def _synthetic_games(n=60, seed=0):
    """Real-ish American-odds/outcome data with a genuine model edge on one
    side, so ROI/CLV aren't trivially all-NaN (n_bets=0) in the test."""
    rng = np.random.default_rng(seed)
    home_win = rng.integers(0, 2, size=n)
    # Market probability close to a coinflip; model is a bit more confident
    # in the direction that actually happened, on average, giving it a real
    # (if modest) edge to place bets on.
    market_home_prob = rng.uniform(0.45, 0.55, size=n)
    model_home_prob = np.clip(market_home_prob + (home_win - 0.5) * 0.15 + rng.normal(0, 0.02, size=n), 0.02, 0.98)
    home_ml_open = np.where(market_home_prob >= 0.5, -120, 110).astype(float)
    away_ml_open = np.where(market_home_prob >= 0.5, 110, -120).astype(float)
    # Closing line drifts slightly toward the actual outcome (typical real-market behavior).
    home_ml_close = home_ml_open + (home_win - 0.5) * 10
    away_ml_close = away_ml_open - (home_win - 0.5) * 10
    sim_mean_total_runs = rng.uniform(4, 12, size=n)
    return pd.DataFrame(
        {
            "home_win": home_win.astype(bool),
            "home_ml_open": home_ml_open,
            "away_ml_open": away_ml_open,
            "home_ml_close": home_ml_close,
            "away_ml_close": away_ml_close,
            "model_home_prob": model_home_prob,
            "sim_mean_total_runs": sim_mean_total_runs,
        }
    )


def test_compute_backtest_metrics_returns_all_expected_keys_and_sane_ranges():
    df = _synthetic_games()
    metrics = compute_backtest_metrics(df, n_resamples=200, seed=1)

    assert metrics["n_games"] == len(df)
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    assert "fraction_of_positives" in metrics["calibration_curve"]
    assert "mean_predicted" in metrics["calibration_curve"]
    assert metrics["n_bets"] > 0  # the synthetic edge should trigger at least some bets
    assert "roi" in metrics and "roi_ci95" in metrics
    assert "mean_clv" in metrics and "clv_ci95" in metrics


def test_compute_backtest_metrics_perfect_predictions_give_zero_brier_and_near_zero_log_loss():
    df = _synthetic_games()
    df = df.copy()
    df["model_home_prob"] = np.where(df["home_win"], 0.999, 0.001)
    metrics = compute_backtest_metrics(df, n_resamples=50, seed=1)
    assert metrics["brier_score"] == pytest.approx(0.0, abs=1e-3)
    assert metrics["log_loss"] < 0.01


def test_compute_pooled_and_tiered_metrics_covers_pooled_and_every_tier_disjointly():
    df = _synthetic_games(n=90)
    df["tier"] = assign_tier(df["sim_mean_total_runs"])
    metrics = compute_pooled_and_tiered_metrics(df, n_resamples=100, seed=2)

    assert set(metrics.keys()) == {"pooled", *TIER_LABELS}
    assert metrics["pooled"]["n_games"] == len(df)
    # Every real game lands in exactly one tier, and every tier's rows sum back to the pooled total.
    tier_total = sum(metrics[label]["n_games"] for label in TIER_LABELS)
    assert tier_total == len(df)


def test_compute_pooled_and_tiered_metrics_handles_an_empty_tier_without_crashing():
    df = _synthetic_games(n=20)
    df = df.copy()
    df["sim_mean_total_runs"] = 8.0  # every game lands in the 7-9 tier -- <7 and >9 are empty.
    df["tier"] = assign_tier(df["sim_mean_total_runs"])
    metrics = compute_pooled_and_tiered_metrics(df, n_resamples=50, seed=3)

    assert metrics["<7"]["n_games"] == 0
    assert metrics[">9"]["n_games"] == 0
    assert metrics["7-9"]["n_games"] == 20
