"""Lightweight, fast running synopsis of an in-progress
season_backtest_simulation.py job: for whatever games have completed so
far, reports mean simulated total runs vs. the REAL total runs for those
same games (not the season-wide average -- an apples-to-apples comparison
on exactly the completed subset), plus simple win-prediction accuracy and
Brier score. No bootstrap CI, no calibration curve, no tier split -- this
is meant to be cheap enough to run at every monitoring check-in during a
long job, not a substitute for season_backtest_metrics.py's full report.

Exists because a gross miscalibration (the 2-out baserunning bug, +35%
run inflation) went undetected for a full ~12-hour run before anyone
looked at aggregate accuracy -- this makes that kind of problem visible
within the first few hundred completed games instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.season_backtest_fixtures import select_season_games_with_betting_lines


def load_partial_results(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        return pd.DataFrame()
    rows = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a trailing line still being written -- skip, next check-in will pick it up
    return pd.DataFrame(rows)


def synopsis(results_path: Path, season: int) -> dict:
    """Real 2025 total runs/actual outcomes come from the same
    games+betting-lines join season_backtest_metrics.py uses -- this can't
    silently drift from what the final report will compute."""
    partial = load_partial_results(results_path)
    if partial.empty:
        return {"n_games": 0}

    games = select_season_games_with_betting_lines(season)
    merged = partial.merge(
        games[["game_pk", "home_score", "away_score", "home_win"]], on="game_pk", how="inner"
    )
    merged["real_total_runs"] = merged["home_score"] + merged["away_score"]

    y_true = merged["home_win"].to_numpy().astype(int)
    p = merged["model_home_prob"].to_numpy(dtype=float)
    brier = float(np.mean((p - y_true) ** 2))
    accuracy = float(np.mean((p > 0.5).astype(int) == y_true))

    return {
        "n_games": len(merged),
        "mean_sim_total_runs": float(merged["sim_mean_total_runs"].mean()),
        "mean_real_total_runs_same_games": float(merged["real_total_runs"].mean()),
        "brier_score": brier,
        "win_accuracy": accuracy,
        "tier_counts": {
            "<7": int((merged["sim_mean_total_runs"] < 7).sum()),
            "7-9": int(((merged["sim_mean_total_runs"] >= 7) & (merged["sim_mean_total_runs"] <= 9)).sum()),
            ">9": int((merged["sim_mean_total_runs"] > 9).sum()),
        },
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast running synopsis of an in-progress season backtest.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--results-path", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    s = synopsis(args.results_path, args.season)
    if s["n_games"] == 0:
        print("No completed games yet.")
        return
    print(f"=== Running synopsis: {s['n_games']} games completed so far ===")
    print(f"Mean simulated total runs:      {s['mean_sim_total_runs']:.2f}")
    print(f"Mean REAL total runs (same games): {s['mean_real_total_runs_same_games']:.2f}")
    print(f"Simulated - real delta:         {s['mean_sim_total_runs'] - s['mean_real_total_runs_same_games']:+.2f}")
    print(f"Win-prediction accuracy:        {s['win_accuracy']:.1%}")
    print(f"Brier score:                    {s['brier_score']:.4f}  (0.25 = coin-flip baseline)")
    print(f"Tier counts so far: <7={s['tier_counts']['<7']}, 7-9={s['tier_counts']['7-9']}, >9={s['tier_counts']['>9']}")


if __name__ == "__main__":
    main()
