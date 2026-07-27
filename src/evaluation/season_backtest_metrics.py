"""Pooled-AND-per-tier metrics for the Phase-11-scoped full-season
simulation-based backtest: Brier score, log-loss, calibration/reliability
curve, closing-line value, and flat-bet ROI with bootstrap confidence
intervals (the last two via src/evaluation/betting_sim.py, unmodified and
reused as-is -- model-agnostic, just needs `home_win` + the four American-
odds columns + a probability series).

Scoring-environment tiers (predicted total <7 / 7-9 / >9 runs) are assigned
from each game's OWN simulated mean total runs (season_backtest_simulation.py's
sim_mean_total_runs), not any data-relative split (terciles etc.) -- fixed
thresholds, locked in before any result is computed, so the boundary can't
drift based on how results happen to look, and stays comparable across
boundary 2, 3, and beyond. This is the check that actually tests whether
this project's three documented Known Limitations (understated contact-
quality differentiation, absent pitcher-batter interaction, overstated TTO
fatigue penalty) contaminate win-probability accuracy specifically in
low-scoring/duel-shaped games -- a pooled-only result would hide exactly
that signal.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

from src.evaluation.betting_sim import DEFAULT_EDGE_THRESHOLD, DEFAULT_N_RESAMPLES, DEFAULT_STAKE, evaluate_betting_strategy
from src.evaluation.season_backtest_fixtures import select_season_games_with_betting_lines

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Locked in before any result is computed (per this validation's own design constraint) -- do not adjust
# based on how results look. <7 / 7-9 / >9 runs, using each game's own simulated mean total.
TIER_LABELS = ["<7", "7-9", ">9"]


def assign_tier(sim_mean_total_runs) -> np.ndarray:
    values = np.asarray(sim_mean_total_runs, dtype=float)
    tiers = np.full(values.shape, "", dtype=object)
    tiers[values < 7] = "<7"
    tiers[(values >= 7) & (values <= 9)] = "7-9"
    tiers[values > 9] = ">9"
    return tiers


def build_analysis_frame(results_path: Path, season: int) -> pd.DataFrame:
    """Joins season_backtest_simulation.py's per-game results (model_home_prob,
    sim_mean_total_runs) against the same real games+betting-lines sample
    season_backtest_fixtures.py defines -- so this can never silently
    analyze a different population than what was actually simulated."""
    results = pd.read_json(results_path, lines=True)
    games = select_season_games_with_betting_lines(season)
    merged = results.merge(
        games[["game_pk", "home_win", "home_ml_open", "away_ml_open", "home_ml_close", "away_ml_close"]],
        on="game_pk", how="inner",
    )
    if len(merged) != len(results):
        logger.warning(
            "%d simulated games did not match the current games+betting-lines sample (dropped from analysis) "
            "-- results_path may be stale relative to the live data.", len(results) - len(merged)
        )
    merged["tier"] = assign_tier(merged["sim_mean_total_runs"])
    return merged


def compute_backtest_metrics(
    df: pd.DataFrame,
    n_bins: int = 10,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    stake: float = DEFAULT_STAKE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int | None = None,
) -> dict:
    """df must have home_win, home_ml_open/close, away_ml_open/close,
    model_home_prob (exactly betting_sim.REQUIRED_GAME_COLUMNS plus
    model_home_prob). Returns n_games, brier_score, log_loss,
    calibration_curve (fraction_of_positives/mean_predicted arrays), plus
    every key evaluate_betting_strategy returns except the raw per-bet
    frame (n_bets, roi, roi_ci95, mean_clv, clv_ci95, ...)."""
    y_true = df["home_win"].to_numpy().astype(int)
    p = df["model_home_prob"].to_numpy(dtype=float)

    brier = float(np.mean((p - y_true) ** 2))
    ll = float(log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))

    n_bins_effective = min(n_bins, max(len(np.unique(p)) // 2, 1))
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=max(n_bins_effective, 1), strategy="uniform")

    betting = evaluate_betting_strategy(df, p, edge_threshold=edge_threshold, stake=stake, n_resamples=n_resamples, seed=seed)
    betting_public = {k: v for k, v in betting.items() if k != "bets"}

    return {
        "n_games": len(df),
        "brier_score": brier,
        "log_loss": ll,
        "calibration_curve": {"fraction_of_positives": frac_pos.tolist(), "mean_predicted": mean_pred.tolist()},
        **betting_public,
    }


def compute_pooled_and_tiered_metrics(df: pd.DataFrame, **kwargs) -> dict[str, dict]:
    """{'pooled': {...}, '<7': {...}, '7-9': {...}, '>9': {...}} -- every
    metric computed identically for the full pool and for each tier, so
    nothing here can silently report pooled-only."""
    out = {"pooled": compute_backtest_metrics(df, **kwargs)}
    for label in TIER_LABELS:
        tier_df = df[df["tier"] == label]
        out[label] = compute_backtest_metrics(tier_df, **kwargs) if len(tier_df) > 0 else {"n_games": 0}
    return out


def _format_report(metrics: dict[str, dict]) -> str:
    rows = []
    header = f"{'segment':<8} {'n_games':>8} {'brier':>8} {'log_loss':>9} {'n_bets':>7} {'roi':>8} {'roi_ci95':>18} {'mean_clv':>9} {'clv_ci95':>18}"
    rows.append(header)
    rows.append("-" * len(header))
    for label in ["pooled", *TIER_LABELS]:
        m = metrics[label]
        if m.get("n_games", 0) == 0:
            rows.append(f"{label:<8} {0:>8}")
            continue
        roi = m.get("roi", float("nan"))
        roi_ci = m.get("roi_ci95", (float("nan"), float("nan")))
        clv = m.get("mean_clv", float("nan"))
        clv_ci = m.get("clv_ci95", (float("nan"), float("nan")))
        rows.append(
            f"{label:<8} {m['n_games']:>8} {m['brier_score']:>8.4f} {m['log_loss']:>9.4f} "
            f"{m.get('n_bets', 0):>7} {roi:>8.3f} [{roi_ci[0]:>7.3f},{roi_ci[1]:>7.3f}] "
            f"{clv:>9.4f} [{clv_ci[0]:>7.4f},{clv_ci[1]:>7.4f}]"
        )
    return "\n".join(rows)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pooled-and-per-tier metrics report for a completed season backtest simulation.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edge-threshold", type=float, default=DEFAULT_EDGE_THRESHOLD)
    parser.add_argument("--stake", type=float, default=DEFAULT_STAKE)
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    df = build_analysis_frame(args.results_path, args.season)
    logger.info("Analysis frame: %d games (%s)", len(df), df["tier"].value_counts().to_dict())

    metrics = compute_pooled_and_tiered_metrics(
        df, edge_threshold=args.edge_threshold, stake=args.stake, n_resamples=args.n_resamples, seed=args.seed
    )

    report = _format_report(metrics)
    logger.info("\n%s", report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "backtest_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (args.output_dir / "backtest_metrics_report.txt").write_text(report)
    logger.info("Wrote %s and %s", args.output_dir / "backtest_metrics.json", args.output_dir / "backtest_metrics_report.txt")


if __name__ == "__main__":
    main()
