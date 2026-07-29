"""ROI/CLV bucketed by edge magnitude (in RUNS, not probability) for
over/under (totals) bets from a completed season backtest -- a different
market than betting_sim.py's moneyline simulation, using the same real
betting-line data (total_open/close + over/under odds, already joined in
by season_backtest_fixtures.select_season_games_with_betting_lines).

Edge here is signed runs: sim_mean_total_runs - total_open. Positive means
the model's own predicted total exceeds the market's line -> bet OVER;
negative -> bet UNDER. Every game gets bet on whichever side the edge
points to (no minimum-edge gate, unlike betting_sim's moneyline
edge_threshold) -- the point of this report is to see how ROI/CLV vary
ACROSS the edge-magnitude spectrum, not to test a single go/no-go
threshold. Bucketed by |edge|:
  - "up to 1 run": |edge| <= 1.0
  - "1.5-2 runs": 1.5 <= |edge| <= 2.0
  - "more than 2 runs": |edge| > 2.0
The 1.0-1.5 exclusive gap isn't covered by either of the user's declared
buckets -- bets landing there are reported separately as "other" rather
than silently dropped or forced into a neighboring bucket, same convention
as season_backtest_edge_moneyline_buckets.py's own gap-handling.

PUSHES (actual_total exactly equals total_open) are real and not rare for
whole-number lines -- excluded from ROI/CLV (stake is returned, nothing was
actually won or lost), but counted and reported separately per bucket so
they're not silently invisible.

CLV simplification, stated explicitly rather than left implicit: computed
as the de-vigged implied probability of the bet SIDE (over or under) at
close minus at open, reusing build_betting_lines.no_vig_home_win_prob
(fully generic despite its name -- just needs two complementary American-
odds series) -- the same price-movement-only convention betting_sim.py
uses for moneylines. This does NOT separately model the total LINE itself
moving (total_close can differ from total_open); a fully rigorous totals
CLV would need a scoring-distribution model to translate a moved line back
onto the original bet's own number, which nothing in this codebase
provides. Treat this CLV as "did the market's price for this side move in
the bettor's favor," not a complete line-and-price CLV.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.build_betting_lines import no_vig_home_win_prob
from src.evaluation.betting_sim import DEFAULT_N_RESAMPLES, DEFAULT_STAKE, _bootstrap_ci, american_odds_profit_if_win
from src.evaluation.season_backtest_metrics import build_analysis_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_TOTALS_COLUMNS = [
    "total_open", "over_open_odds", "under_open_odds",
    "total_close", "over_close_odds", "under_close_odds",
    "home_score", "away_score",
]

# (label, predicate on |edge| in runs). See module docstring for the 1.0-1.5 gap.
EDGE_BUCKETS = [
    ("up to 1 run", lambda e: e <= 1.0),
    ("1.5-2 runs", lambda e: (e >= 1.5) & (e <= 2.0)),
    ("more than 2 runs", lambda e: e > 2.0),
]


def simulate_totals_betting(
    games: pd.DataFrame, sim_mean_total_runs, stake: float = DEFAULT_STAKE, bias_correction: float = 0.0
) -> pd.DataFrame:
    """One row per candidate game, whether or not a bet was placed. See
    module docstring for edge/side/push/CLV definitions. Returned columns:
    edge, side (None where edge==0 exactly), placed, is_push, odds, stake,
    profit, clv (last four NaN/0 where no bet was placed; profit is 0.0,
    not NaN, on a push, since a push is a placed-but-non-P&L bet).

    bias_correction: subtracted from sim_mean_total_runs before computing
    edge/side (i.e. adjusted_prediction = sim_mean_total_runs -
    bias_correction) -- e.g. 0.33 to test betting against the model's
    predicted total net of this backtest's own measured mean simulated-vs-
    real runs/game gap (see the full-run synopsis: 9.18 sim vs 8.85 real =
    +0.33). Does NOT touch total_open, the real actual_total used for
    win/push determination, or the real odds/CLV inputs -- only the
    prediction driving the betting DECISION is adjusted. 0.0 (default)
    reproduces the original, uncorrected behavior exactly."""
    missing = [c for c in REQUIRED_TOTALS_COLUMNS if c not in games.columns]
    if missing:
        raise ValueError(f"games is missing required column(s): {missing}")

    games = games.reset_index(drop=True)
    sim_mean_total_runs = np.asarray(sim_mean_total_runs, dtype=float) - bias_correction
    if len(sim_mean_total_runs) != len(games):
        raise ValueError(f"sim_mean_total_runs length ({len(sim_mean_total_runs)}) != games length ({len(games)})")

    total_open = games["total_open"].to_numpy(dtype=float)
    edge = sim_mean_total_runs - total_open

    bet_over = edge > 0
    bet_under = edge < 0
    placed = bet_over | bet_under
    side = np.where(bet_over, "over", np.where(bet_under, "under", None))
    bet_odds = np.where(bet_over, games["over_open_odds"].to_numpy(dtype=float), games["under_open_odds"].to_numpy(dtype=float))

    actual_total = games["home_score"].to_numpy(dtype=float) + games["away_score"].to_numpy(dtype=float)
    is_push = actual_total == total_open
    win = np.where(bet_over, actual_total > total_open, actual_total < total_open)
    profit = np.where(is_push, 0.0, np.where(win, stake * american_odds_profit_if_win(bet_odds), -stake))

    open_over_prob = no_vig_home_win_prob(games["over_open_odds"], games["under_open_odds"])
    close_over_prob = no_vig_home_win_prob(games["over_close_odds"], games["under_close_odds"])
    open_prob_bet_side = np.where(bet_over, open_over_prob, 1 - open_over_prob)
    close_prob_bet_side = np.where(bet_over, close_over_prob, 1 - close_over_prob)
    clv = close_prob_bet_side - open_prob_bet_side

    return pd.DataFrame(
        {
            "edge": edge,
            "side": side,
            "placed": placed,
            "is_push": np.where(placed, is_push, False),
            "odds": np.where(placed, bet_odds, np.nan),
            "stake": np.where(placed, stake, 0.0),
            "profit": np.where(placed, profit, np.nan),
            "clv": np.where(placed, clv, np.nan),
        }
    )


def assign_edge_bucket(edge_magnitude) -> np.ndarray:
    edge_magnitude = np.asarray(edge_magnitude, dtype=float)
    labels = np.full(edge_magnitude.shape, "other", dtype=object)
    for label, predicate in EDGE_BUCKETS:
        labels[predicate(edge_magnitude)] = label
    return labels


def _bucket_stats(bucket_placed: pd.DataFrame, bucket_non_push: pd.DataFrame, stake: float, n_resamples: int, seed: int | None) -> dict:
    n_pushes = int(bucket_placed["is_push"].sum())
    n_bets = len(bucket_non_push)
    if n_bets == 0:
        return {
            "n_bets": 0, "n_pushes": n_pushes, "roi": float("nan"),
            "roi_ci_lo": float("nan"), "roi_ci_hi": float("nan"), "mean_clv": float("nan"),
            "clv_ci_lo": float("nan"), "clv_ci_hi": float("nan"),
        }
    profits = bucket_non_push["profit"].to_numpy()
    clvs = bucket_non_push["clv"].to_numpy()
    roi = float(profits.sum() / (n_bets * stake))
    roi_ci = _bootstrap_ci(profits, lambda p_: float(p_.sum() / (len(p_) * stake)), n_resamples, seed)
    clv_ci = _bootstrap_ci(clvs, lambda c: float(c.mean()), n_resamples, seed)
    return {
        "n_bets": n_bets, "n_pushes": n_pushes, "roi": roi,
        "roi_ci_lo": roi_ci[0], "roi_ci_hi": roi_ci[1], "mean_clv": float(clvs.mean()),
        "clv_ci_lo": clv_ci[0], "clv_ci_hi": clv_ci[1],
    }


def compute_totals_edge_report(
    df: pd.DataFrame,
    sim_mean_total_runs_col: str = "sim_mean_total_runs",
    stake: float = DEFAULT_STAKE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int | None = None,
    bias_correction: float = 0.0,
) -> pd.DataFrame:
    """df must have REQUIRED_TOTALS_COLUMNS plus sim_mean_total_runs_col.
    Returns one row per edge bucket, PLUS "other" (the 1.0-1.5 gap -- see
    module docstring, reported explicitly rather than silently folded into
    only the "ALL" row) and "ALL" (every placed bet combined), with n_bets
    (non-push), n_pushes, roi, roi_ci95, mean_clv, clv_ci95. See
    simulate_totals_betting for bias_correction's meaning."""
    bets = simulate_totals_betting(df, df[sim_mean_total_runs_col], stake=stake, bias_correction=bias_correction)
    placed = bets[bets["placed"]].copy()
    placed["edge_bucket"] = assign_edge_bucket(placed["edge"].abs())
    non_push = placed[~placed["is_push"]]

    bucket_labels = [label for label, _ in EDGE_BUCKETS]
    rows = []
    for label in [*bucket_labels, "other", "ALL"]:
        bucket_placed = placed if label == "ALL" else placed[placed["edge_bucket"] == label]
        bucket_non_push = non_push if label == "ALL" else non_push[non_push["edge_bucket"] == label]
        rows.append({"edge_bucket": label, **_bucket_stats(bucket_placed, bucket_non_push, stake, n_resamples, seed)})
    return pd.DataFrame(rows)


def compute_totals_edge_side_report(
    df: pd.DataFrame,
    sim_mean_total_runs_col: str = "sim_mean_total_runs",
    stake: float = DEFAULT_STAKE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int | None = None,
    bias_correction: float = 0.0,
) -> pd.DataFrame:
    """Same as compute_totals_edge_report, crossed with bet side (over/
    under) -- one row per (edge_bucket, side) pair, for edge_bucket in the
    3 declared buckets + "other", each split into "over" and "under", plus
    an "ALL" edge_bucket row per side (every edge magnitude combined, that
    side only) so the side-only effect is visible too. See
    simulate_totals_betting for bias_correction's meaning."""
    bets = simulate_totals_betting(df, df[sim_mean_total_runs_col], stake=stake, bias_correction=bias_correction)
    placed = bets[bets["placed"]].copy()
    placed["edge_bucket"] = assign_edge_bucket(placed["edge"].abs())
    non_push = placed[~placed["is_push"]]

    bucket_labels = [label for label, _ in EDGE_BUCKETS]
    rows = []
    for edge_label in [*bucket_labels, "other", "ALL"]:
        edge_placed = placed if edge_label == "ALL" else placed[placed["edge_bucket"] == edge_label]
        edge_non_push = non_push if edge_label == "ALL" else non_push[non_push["edge_bucket"] == edge_label]
        for side in ["over", "under"]:
            side_placed = edge_placed[edge_placed["side"] == side]
            side_non_push = edge_non_push[edge_non_push["side"] == side]
            rows.append(
                {"edge_bucket": edge_label, "side": side, **_bucket_stats(side_placed, side_non_push, stake, n_resamples, seed)}
            )
    return pd.DataFrame(rows)


def _format_report(report: pd.DataFrame) -> str:
    header = f"{'edge_bucket':<18} {'n_bets':>7} {'n_pushes':>9} {'roi':>8} {'roi_ci95':>18} {'mean_clv':>9} {'clv_ci95':>18}"
    lines = [header, "-" * len(header)]
    for _, row in report.iterrows():
        if row["n_bets"] == 0:
            lines.append(f"{row['edge_bucket']:<18} {0:>7} {int(row['n_pushes']):>9}")
            continue
        lines.append(
            f"{row['edge_bucket']:<18} {int(row['n_bets']):>7} {int(row['n_pushes']):>9} {row['roi']:>8.3f} "
            f"[{row['roi_ci_lo']:>7.3f},{row['roi_ci_hi']:>7.3f}] {row['mean_clv']:>9.4f} "
            f"[{row['clv_ci_lo']:>7.4f},{row['clv_ci_hi']:>7.4f}]"
        )
    return "\n".join(lines)


def _format_side_report(report: pd.DataFrame) -> str:
    header = f"{'edge_bucket':<18} {'side':<6} {'n_bets':>7} {'n_pushes':>9} {'roi':>8} {'roi_ci95':>18} {'mean_clv':>9} {'clv_ci95':>18}"
    lines = [header, "-" * len(header)]
    for _, row in report.iterrows():
        if row["n_bets"] == 0:
            lines.append(f"{row['edge_bucket']:<18} {row['side']:<6} {0:>7} {int(row['n_pushes']):>9}")
            continue
        lines.append(
            f"{row['edge_bucket']:<18} {row['side']:<6} {int(row['n_bets']):>7} {int(row['n_pushes']):>9} {row['roi']:>8.3f} "
            f"[{row['roi_ci_lo']:>7.3f},{row['roi_ci_hi']:>7.3f}] {row['mean_clv']:>9.4f} "
            f"[{row['clv_ci_lo']:>7.4f},{row['clv_ci_hi']:>7.4f}]"
        )
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROI/CLV bucketed by run-edge magnitude (and optionally side) for over/under bets from a completed season backtest.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stake", type=float, default=DEFAULT_STAKE)
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bias-correction", type=float, default=0.0,
        help="Subtracted from sim_mean_total_runs before computing edge/side for both reports -- see "
        "simulate_totals_betting's docstring. 0.0 (default) reproduces the original, uncorrected behavior.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    df = build_analysis_frame(args.results_path, args.season)
    logger.info("Analysis frame: %d games (bias_correction=%s)", len(df), args.bias_correction)

    report = compute_totals_edge_report(
        df, stake=args.stake, n_resamples=args.n_resamples, seed=args.seed, bias_correction=args.bias_correction
    )
    text_report = _format_report(report)
    logger.info("\n%s", text_report)

    side_report = compute_totals_edge_side_report(
        df, stake=args.stake, n_resamples=args.n_resamples, seed=args.seed, bias_correction=args.bias_correction
    )
    text_side_report = _format_side_report(side_report)
    logger.info("\n%s", text_side_report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_json(args.output_dir / "totals_edge_buckets.json", orient="records", indent=2)
    (args.output_dir / "totals_edge_buckets_report.txt").write_text(text_report)
    side_report.to_json(args.output_dir / "totals_edge_side_buckets.json", orient="records", indent=2)
    (args.output_dir / "totals_edge_side_buckets_report.txt").write_text(text_side_report)
    logger.info(
        "Wrote %s, %s, %s, %s",
        args.output_dir / "totals_edge_buckets.json", args.output_dir / "totals_edge_buckets_report.txt",
        args.output_dir / "totals_edge_side_buckets.json", args.output_dir / "totals_edge_side_buckets_report.txt",
    )


if __name__ == "__main__":
    main()
