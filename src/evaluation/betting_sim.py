"""Flat-stake moneyline betting simulation against real Vegas odds.

Converts a model's win probabilities into a concrete profit/loss history by
betting a flat stake whenever the model's probability disagrees with the
market's own de-vigged probability by more than a configurable edge, at the
odds actually available at decision time (the *opening* line -- the closing
line is only known after a bet would have been placed, so it can't be the
price a bet is decided against). This is a genuinely different question from
accuracy or Brier score: a model can be well-calibrated overall and still
have no betting edge on the specific games where it disagrees with the
market, and a model with mediocre overall accuracy can still show a real
edge if its disagreements with the market are concentrated in its favor.

Two things are reported beyond raw ROI:
  - A paired bootstrap confidence interval on ROI (resampling the placed
    bets, not every candidate game), the same style as bootstrap_compare.py
    -- so "this fold's ROI was +6%" can be told apart from "the model placed
    9 bets and got lucky."
  - Closing line value (CLV): for every bet actually placed, whether the
    de-vigged price at decision time (open) was better than what the market
    settled on by close, on that same side. Positive CLV -- the bet was made
    at a price the market later moved away from, in the bettor's favor --
    is the standard sports-betting signal of a genuine pricing edge that's
    independent of whether any individual bet actually won; a bettor can be
    profitable-by-CLV over a sample too small for ROI's bootstrap CI to have
    resolved significance yet.

Nothing here trains anything or knows about GamePredictor/LR baselines/
ensembles -- see src/evaluation/walk_forward_betting.py for the script that
supplies real per-fold predictions and betting_lines odds as input.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.data.build_betting_lines import no_vig_home_win_prob

DEFAULT_EDGE_THRESHOLD = 0.02
DEFAULT_STAKE = 1.0
DEFAULT_N_RESAMPLES = 1000

REQUIRED_GAME_COLUMNS = ["home_win", "home_ml_open", "away_ml_open", "home_ml_close", "away_ml_close"]


def american_odds_profit_if_win(odds) -> np.ndarray:
    """Net profit per 1 unit staked if a bet at `odds` wins (American odds):
    underdog (positive odds) pays odds/100, favorite (negative odds) pays
    100/|odds|. Excludes the stake itself -- a loss is a separate -stake."""
    odds = np.asarray(odds, dtype=float)
    return np.where(odds > 0, odds / 100.0, 100.0 / np.abs(odds))


def simulate_flat_stake_betting(
    games: pd.DataFrame,
    model_home_prob,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    stake: float = DEFAULT_STAKE,
) -> pd.DataFrame:
    """One row per candidate game, whether or not a bet was actually placed.

    `games` must have `home_win` (bool/0-1) plus `home_ml_open`/`away_ml_open`/
    `home_ml_close`/`away_ml_close` (American odds) -- see
    data/processed/betting_lines. `model_home_prob` is the model's predicted
    P(home team wins), aligned index-for-index with `games`.

    Both sides' edge against the market can never be positive at once (the
    de-vigged market probabilities and the model's probability are each a
    complementary pair summing to 1), so a bet is placed on whichever side
    clears `edge_threshold`: home if model_home_prob exceeds the open market's
    home probability by more than the threshold, away if it falls short by
    more than the threshold, otherwise no bet.

    Returned columns: model_home_prob, open_home_prob, close_home_prob,
    edge_home, side (None where no bet was placed), placed, odds, stake,
    profit, clv -- the last four NaN/0 where no bet was placed.
    """
    missing = [c for c in REQUIRED_GAME_COLUMNS if c not in games.columns]
    if missing:
        raise ValueError(f"games is missing required column(s): {missing}")

    games = games.reset_index(drop=True)
    model_home_prob = np.asarray(model_home_prob, dtype=float)
    if len(model_home_prob) != len(games):
        raise ValueError(f"model_home_prob length ({len(model_home_prob)}) != games length ({len(games)})")

    open_home_prob = no_vig_home_win_prob(games["home_ml_open"], games["away_ml_open"])
    close_home_prob = no_vig_home_win_prob(games["home_ml_close"], games["away_ml_close"])
    edge_home = model_home_prob - open_home_prob

    bet_home = edge_home > edge_threshold
    bet_away = edge_home < -edge_threshold
    placed = bet_home | bet_away

    side = np.where(bet_home, "home", np.where(bet_away, "away", None))
    bet_odds = np.where(bet_home, games["home_ml_open"].to_numpy(dtype=float), games["away_ml_open"].to_numpy(dtype=float))
    home_win = games["home_win"].to_numpy().astype(bool)
    win = np.where(bet_home, home_win, ~home_win)
    open_prob_bet_side = np.where(bet_home, open_home_prob, 1 - open_home_prob)
    close_prob_bet_side = np.where(bet_home, close_home_prob, 1 - close_home_prob)

    profit = np.where(win, stake * american_odds_profit_if_win(bet_odds), -stake)
    clv = close_prob_bet_side - open_prob_bet_side

    return pd.DataFrame(
        {
            "model_home_prob": model_home_prob,
            "open_home_prob": open_home_prob,
            "close_home_prob": close_home_prob,
            "edge_home": edge_home,
            "side": side,
            "placed": placed,
            "odds": np.where(placed, bet_odds, np.nan),
            "stake": np.where(placed, stake, 0.0),
            "profit": np.where(placed, profit, np.nan),
            "clv": np.where(placed, clv, np.nan),
        }
    )


def _bootstrap_ci(
    values: np.ndarray, statistic: Callable[[np.ndarray], float], n_resamples: int, seed: int | None
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        draws[i] = statistic(values[idx])
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def evaluate_betting_strategy(
    games: pd.DataFrame,
    model_home_prob,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    stake: float = DEFAULT_STAKE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int | None = None,
) -> dict:
    """Runs `simulate_flat_stake_betting` and summarizes it: total ROI, bet
    count, a paired bootstrap 95% CI on ROI (resampling only the placed
    bets), and mean closing-line value with its own bootstrap 95% CI.

    Returns a dict with n_candidates, n_bets, total_staked, total_profit,
    roi, roi_ci95, mean_clv, clv_ci95, and `bets` (the full per-game frame
    from simulate_flat_stake_betting, for callers that want per-bet detail).
    """
    bets = simulate_flat_stake_betting(games, model_home_prob, edge_threshold, stake)
    placed = bets[bets["placed"]]
    n_bets = len(placed)

    if n_bets == 0:
        return {
            "n_candidates": len(bets),
            "n_bets": 0,
            "total_staked": 0.0,
            "total_profit": 0.0,
            "roi": float("nan"),
            "roi_ci95": (float("nan"), float("nan")),
            "mean_clv": float("nan"),
            "clv_ci95": (float("nan"), float("nan")),
            "bets": bets,
        }

    profits = placed["profit"].to_numpy()
    clvs = placed["clv"].to_numpy()
    total_staked = float(n_bets * stake)
    total_profit = float(profits.sum())

    roi_ci = _bootstrap_ci(profits, lambda p: float(p.sum() / (len(p) * stake)), n_resamples, seed)
    clv_ci = _bootstrap_ci(clvs, lambda c: float(c.mean()), n_resamples, seed)

    return {
        "n_candidates": len(bets),
        "n_bets": n_bets,
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": total_profit / total_staked,
        "roi_ci95": roi_ci,
        "mean_clv": float(clvs.mean()),
        "clv_ci95": clv_ci,
        "bets": bets,
    }
