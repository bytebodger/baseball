"""ROI/CLV bucketed by (edge threshold x moneyline range) for a completed
season backtest's moneyline bets -- a finer-grained cut than
season_backtest_metrics.py's pooled/per-scoring-tier report, answering a
different question: does the model's betting edge hold up specifically at
higher-conviction edge thresholds, and does it vary by how big a favorite/
underdog the bet side is.

Edge thresholds are cumulative/nested ("2%+" includes every bet "4%+" and
"6%+" also include, not a disjoint 2-4%/4-6%/6%+ binning) -- the natural
question this answers is "if I only bet when my edge is at least X%, how
does ROI/CLV look," for a few values of X, not "how much of my action was
in each narrow edge band."

Moneyline buckets are on the ACTUAL bet side's odds (home_ml_open if the
model bet home, away_ml_open if it bet away) -- the price actually being
laid, not some other reference line. Reuses betting_sim.py's
simulate_flat_stake_betting and its own bootstrap CI helper directly, so
this can never silently drift from the pooled/per-tier report's own ROI/CLV
methodology.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.betting_sim import DEFAULT_N_RESAMPLES, DEFAULT_STAKE, _bootstrap_ci, simulate_flat_stake_betting
from src.evaluation.season_backtest_metrics import build_analysis_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Cumulative/nested -- "4%+" is a subset of "2%+", not a disjoint band. See module docstring.
EDGE_THRESHOLDS = [("2%+", 0.02), ("4%+", 0.04), ("6%+", 0.06)]

# (label, predicate) -- predicate takes the odds array, returns a boolean mask.
# The trivial (-100, 100) exclusive gap isn't a real American-odds region (see module docstring) --
# anything landing there (or exactly -100) is reported separately as "other" rather than silently dropped.
MONEYLINE_BUCKETS = [
    ("< -200", lambda o: o < -200),
    ("-151 to -200", lambda o: (o >= -200) & (o <= -151)),
    ("-101 to -150", lambda o: (o >= -150) & (o <= -101)),
    ("+100 to +150", lambda o: (o >= 100) & (o <= 150)),
    ("+151 to +200", lambda o: (o >= 151) & (o <= 200)),
    ("> +200", lambda o: o > 200),
]


def assign_moneyline_bucket(odds) -> np.ndarray:
    odds = np.asarray(odds, dtype=float)
    labels = np.full(odds.shape, "other", dtype=object)
    for label, predicate in MONEYLINE_BUCKETS:
        labels[predicate(odds)] = label
    return labels


def compute_edge_moneyline_report(
    df: pd.DataFrame,
    edge_thresholds: list[tuple[str, float]] = EDGE_THRESHOLDS,
    moneyline_buckets: list[tuple[str, object]] = MONEYLINE_BUCKETS,
    stake: float = DEFAULT_STAKE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int | None = None,
) -> pd.DataFrame:
    """df: same shape season_backtest_metrics.build_analysis_frame produces
    (home_win, home_ml_open/close, away_ml_open/close, model_home_prob).
    Returns one row per (edge_threshold, moneyline_bucket) cell, plus an
    "ALL" moneyline row per edge threshold (every bucket combined, so a
    reader can see the edge-threshold-only effect too)."""
    p = df["model_home_prob"].to_numpy(dtype=float)
    moneyline_labels = [label for label, _ in moneyline_buckets]

    rows = []
    for edge_label, edge_threshold in edge_thresholds:
        bets = simulate_flat_stake_betting(df, p, edge_threshold=edge_threshold, stake=stake)
        placed = bets[bets["placed"]].copy()
        placed["moneyline_bucket"] = assign_moneyline_bucket(placed["odds"])

        for ml_label in [*moneyline_labels, "ALL"]:
            subset = placed if ml_label == "ALL" else placed[placed["moneyline_bucket"] == ml_label]
            n_bets = len(subset)
            if n_bets == 0:
                rows.append(
                    {"edge_threshold": edge_label, "moneyline_bucket": ml_label, "n_bets": 0, "roi": float("nan"),
                     "roi_ci_lo": float("nan"), "roi_ci_hi": float("nan"), "mean_clv": float("nan"),
                     "clv_ci_lo": float("nan"), "clv_ci_hi": float("nan")}
                )
                continue
            profits = subset["profit"].to_numpy()
            clvs = subset["clv"].to_numpy()
            roi = float(profits.sum() / (n_bets * stake))
            roi_ci = _bootstrap_ci(profits, lambda p_: float(p_.sum() / (len(p_) * stake)), n_resamples, seed)
            clv_ci = _bootstrap_ci(clvs, lambda c: float(c.mean()), n_resamples, seed)
            rows.append(
                {
                    "edge_threshold": edge_label, "moneyline_bucket": ml_label, "n_bets": n_bets, "roi": roi,
                    "roi_ci_lo": roi_ci[0], "roi_ci_hi": roi_ci[1], "mean_clv": float(clvs.mean()),
                    "clv_ci_lo": clv_ci[0], "clv_ci_hi": clv_ci[1],
                }
            )
    return pd.DataFrame(rows)


def _format_report(report: pd.DataFrame) -> str:
    lines = []
    header = f"{'edge':<7} {'moneyline':<14} {'n_bets':>7} {'roi':>8} {'roi_ci95':>18} {'mean_clv':>9} {'clv_ci95':>18}"
    for edge_label, _ in EDGE_THRESHOLDS:
        lines.append(f"=== edge >= {edge_label} ===")
        lines.append(header)
        lines.append("-" * len(header))
        block = report[report["edge_threshold"] == edge_label]
        for _, row in block.iterrows():
            if row["n_bets"] == 0:
                lines.append(f"{edge_label:<7} {row['moneyline_bucket']:<14} {0:>7}")
                continue
            lines.append(
                f"{edge_label:<7} {row['moneyline_bucket']:<14} {int(row['n_bets']):>7} {row['roi']:>8.3f} "
                f"[{row['roi_ci_lo']:>7.3f},{row['roi_ci_hi']:>7.3f}] {row['mean_clv']:>9.4f} "
                f"[{row['clv_ci_lo']:>7.4f},{row['clv_ci_hi']:>7.4f}]"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROI/CLV bucketed by edge threshold x moneyline range for a completed season backtest.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stake", type=float, default=DEFAULT_STAKE)
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    df = build_analysis_frame(args.results_path, args.season)
    logger.info("Analysis frame: %d games", len(df))

    report = compute_edge_moneyline_report(df, stake=args.stake, n_resamples=args.n_resamples, seed=args.seed)
    text_report = _format_report(report)
    logger.info("\n%s", text_report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_json(args.output_dir / "edge_moneyline_buckets.json", orient="records", indent=2)
    (args.output_dir / "edge_moneyline_buckets_report.txt").write_text(text_report)
    logger.info("Wrote %s and %s", args.output_dir / "edge_moneyline_buckets.json", args.output_dir / "edge_moneyline_buckets_report.txt")


if __name__ == "__main__":
    main()
