import numpy as np
import pandas as pd
import pytest

from src.evaluation.betting_sim import (
    american_odds_profit_if_win,
    evaluate_betting_strategy,
    simulate_flat_stake_betting,
)


def _games(rows):
    """rows: list of (home_win, home_ml_open, away_ml_open, home_ml_close, away_ml_close)."""
    cols = ["home_win", "home_ml_open", "away_ml_open", "home_ml_close", "away_ml_close"]
    return pd.DataFrame(rows, columns=cols)


def test_american_odds_profit_if_win_favorite_and_underdog():
    profit = american_odds_profit_if_win(np.array([-110, 150]))
    assert profit[0] == pytest.approx(100.0 / 110.0)
    assert profit[1] == pytest.approx(1.5)


def test_no_bet_when_edge_is_within_threshold():
    # open market is -110/-110 (pick 'em, 0.5/0.5 no-vig); model agrees almost exactly
    games = _games([(1, -110, -110, -110, -110)])
    bets = simulate_flat_stake_betting(games, model_home_prob=[0.505], edge_threshold=0.02)
    assert bets.loc[0, "side"] is None
    assert not bets.loc[0, "placed"]
    assert np.isnan(bets.loc[0, "profit"])


def test_bets_home_when_model_prob_exceeds_open_market_by_more_than_threshold():
    # open market home implied prob = 0.5 (pick 'em); model says 0.6 -> edge = 0.10
    games = _games([(1, -110, -110, -110, -110)])
    bets = simulate_flat_stake_betting(games, model_home_prob=[0.60], edge_threshold=0.02, stake=1.0)
    assert bets.loc[0, "side"] == "home"
    assert bets.loc[0, "placed"]
    # home_win=1 -> the home bet wins, at -110 -> profit = 100/110
    assert bets.loc[0, "profit"] == pytest.approx(100.0 / 110.0)


def test_bets_away_when_model_prob_is_below_open_market_by_more_than_threshold():
    games = _games([(1, -110, -110, -110, -110)])  # home wins, but we'll bet away
    bets = simulate_flat_stake_betting(games, model_home_prob=[0.40], edge_threshold=0.02, stake=1.0)
    assert bets.loc[0, "side"] == "away"
    assert bets.loc[0, "placed"]
    # bet away, but home actually won -> away bet loses -> profit = -stake
    assert bets.loc[0, "profit"] == pytest.approx(-1.0)


def test_clv_is_positive_when_market_moves_toward_the_bet_side_by_close():
    # open: pick 'em (0.5/0.5 home/away). close: home shortens to a favorite,
    # i.e. the market moved toward home after we bet home -> positive CLV for a home bet.
    games = _games([(1, -110, -110, -150, 130)])
    bets = simulate_flat_stake_betting(games, model_home_prob=[0.60], edge_threshold=0.02)
    assert bets.loc[0, "side"] == "home"
    open_prob = bets.loc[0, "open_home_prob"]
    close_prob = bets.loc[0, "close_home_prob"]
    assert close_prob > open_prob  # market shortened home by close
    assert bets.loc[0, "clv"] == pytest.approx(close_prob - open_prob)
    assert bets.loc[0, "clv"] > 0


def test_evaluate_betting_strategy_computes_roi_and_bet_count():
    # three games, all with a clear home edge and -110/-110 pick'em pricing;
    # 2 home wins, 1 home loss.
    games = _games(
        [
            (1, -110, -110, -110, -110),
            (1, -110, -110, -110, -110),
            (0, -110, -110, -110, -110),
        ]
    )
    result = evaluate_betting_strategy(games, model_home_prob=[0.6, 0.6, 0.6], edge_threshold=0.02, n_resamples=200, seed=0)

    assert result["n_candidates"] == 3
    assert result["n_bets"] == 3
    win_profit = 100.0 / 110.0
    expected_total_profit = win_profit + win_profit - 1.0
    assert result["total_profit"] == pytest.approx(expected_total_profit)
    assert result["roi"] == pytest.approx(expected_total_profit / 3.0)
    lo, hi = result["roi_ci95"]
    assert lo <= result["roi"] <= hi


def test_evaluate_betting_strategy_handles_zero_bets_without_crashing():
    games = _games([(1, -110, -110, -110, -110)])
    # model agrees almost exactly with the market -> no edge -> no bets
    result = evaluate_betting_strategy(games, model_home_prob=[0.5], edge_threshold=0.02)

    assert result["n_bets"] == 0
    assert result["n_candidates"] == 1
    assert np.isnan(result["roi"])
    assert np.isnan(result["roi_ci95"][0]) and np.isnan(result["roi_ci95"][1])
    assert np.isnan(result["mean_clv"])


def test_mismatched_lengths_raise():
    games = _games([(1, -110, -110, -110, -110), (0, -110, -110, -110, -110)])
    with pytest.raises(ValueError):
        simulate_flat_stake_betting(games, model_home_prob=[0.6])


def test_missing_required_column_raises():
    games = pd.DataFrame({"home_win": [1], "home_ml_open": [-110]})
    with pytest.raises(ValueError, match="missing required column"):
        simulate_flat_stake_betting(games, model_home_prob=[0.6])
