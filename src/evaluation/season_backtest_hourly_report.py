"""Standing hourly status report for an in-progress season_backtest_simulation.py
job (user request, 2026-07-29): games completed/remaining, win-prediction
accuracy, mean simulated runs/game, and aggregate CLV against real Vegas
moneyline odds for whatever games have completed so far.

Reuses season_backtest_progress_synopsis.py's partial-results loader and
season_backtest_fixtures.py's real games+betting-lines join (never drifts
from what the final report computes), plus betting_sim.py's established
CLV methodology (same edge-threshold/no-vig convention used everywhere else
in this project) -- not a new metric definition invented for this report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.betting_sim import DEFAULT_EDGE_THRESHOLD, evaluate_betting_strategy
from src.evaluation.season_backtest_fixtures import select_season_games_with_betting_lines
from src.evaluation.season_backtest_progress_synopsis import load_partial_results
from src.resumable_job import read_progress


def hourly_report(results_path: Path, progress_path: Path, season: int) -> dict:
    progress = read_progress(progress_path)
    partial = load_partial_results(results_path)
    if partial.empty:
        return {"n_completed": 0, "n_total": progress.total if progress else None}

    games = select_season_games_with_betting_lines(season)
    merged = partial.merge(games, on="game_pk", how="inner", suffixes=("", "_line"))

    y_true = merged["home_win"].to_numpy().astype(int)
    p = merged["model_home_prob"].to_numpy(dtype=float)
    win_accuracy = float(np.mean((p > 0.5).astype(int) == y_true))

    bet_eval = evaluate_betting_strategy(merged, p, edge_threshold=DEFAULT_EDGE_THRESHOLD)

    return {
        "n_completed": progress.completed if progress else len(partial),
        "n_total": progress.total if progress else None,
        "n_remaining": progress.remaining if progress else None,
        "n_joined_with_lines": len(merged),
        "win_accuracy": win_accuracy,
        "mean_sim_total_runs": float(merged["sim_mean_total_runs"].mean()),
        "n_bets_placed": bet_eval["n_bets"],
        "mean_clv": bet_eval["mean_clv"],
        "clv_ci95": bet_eval["clv_ci95"],
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hourly status report for an in-progress season backtest.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--progress-path", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    r = hourly_report(args.results_path, args.progress_path, args.season)
    if r["n_completed"] == 0:
        print("No completed games yet.")
        return
    print(f"=== Hourly status: {r['n_completed']}/{r['n_total']} games simulated, {r['n_remaining']} remaining ===")
    print(f"Win-prediction accuracy (of {r['n_joined_with_lines']} completed games with real betting lines): {r['win_accuracy']:.1%}")
    print(f"Mean simulated runs/game: {r['mean_sim_total_runs']:.2f}")
    print(
        f"Aggregate CLV: {r['mean_clv']:+.4f} (95% CI {r['clv_ci95'][0]:+.4f} to {r['clv_ci95'][1]:+.4f}), "
        f"over {r['n_bets_placed']}/{r['n_joined_with_lines']} games where the model's edge exceeded the "
        f"{DEFAULT_EDGE_THRESHOLD:.0%} betting threshold"
    )


if __name__ == "__main__":
    main()
