"""Runs the flat-stake betting simulation (src/evaluation/betting_sim.py)
against GamePredictor, the ERA/wRC+ logistic-regression baseline, and their
stacked ensemble, across all 6 walk-forward folds (see
src/inference/walk_forward_backtest.py).

This answers a different question than that script's accuracy/Brier
backtest: not "is this model's win probability closer to the truth than
baseline X," but "would betting flat stakes on this model's disagreements
with the market, at the real Vegas moneyline, have actually made money" --
a model can be well-calibrated overall and still have no edge on the
specific games where it disagrees with the market enough to bet.

Reuses each fold's already-trained GamePredictor checkpoint (no retraining):
- The LR baseline is refit on that fold's train+val games, same convention
  as walk_forward_backtest.py's backtest_fold.
- A stacked logistic-regression ensemble ([GamePredictor prob, LR prob] ->
  home_win) is fit on that fold's *validation*-season predictions --
  genuinely out-of-sample for GamePredictor, the same convention
  backtest.py's generate_predictions uses for the fixed split -- then
  applied to the fold's test season.
- All three methods' test-season predictions are joined to
  data/processed/betting_lines (open + close moneylines, matched games
  only) and run through betting_sim.py.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from src.data.game_dataset import BATTER_APPEARANCES_DIR, DEFAULT_BULLPEN_WINDOW_DAYS, DEFAULT_MAX_LINEUP_SIZE, GAMES_DIR, PITCHER_APPEARANCES_DIR
from src.data.statcast_common import PROCESSED_DATA_DIR, RAW_DATA_DIR, read_partitioned
from src.device import DEFAULT_DEVICE, resolve_device
from src.evaluation.betting_sim import DEFAULT_EDGE_THRESHOLD, DEFAULT_N_RESAMPLES, DEFAULT_STAKE, evaluate_betting_strategy
from src.inference.backtest import (
    ENSEMBLE_METHOD,
    FEATURE_COLUMNS,
    GAME_PREDICTOR_METHOD,
    LOGISTIC_REGRESSION_METHOD,
    apply_stacking_ensemble,
    build_baseline_features,
    fit_logistic_regression_baseline,
    fit_stacking_ensemble,
    load_trained_system,
    run_model_inference,
)
from src.inference.walk_forward_backtest import (
    BETTING_LINES_DIR,
    DEFAULT_SEASON_RANGE,
    DEFAULT_SEQUENCE_CACHE_DIR,
    DEFAULT_TEST_YEARS,
    DEFAULT_TRAIN_YEARS,
    DEFAULT_VAL_YEARS,
    Fold,
    _load_fold_games,
    generate_folds,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BETTING_LINE_COLUMNS = ["game_pk", "home_ml_open", "away_ml_open", "home_ml_close", "away_ml_close"]


def evaluate_fold_betting(
    fold: Fold,
    checkpoint_path: Path,
    games: pd.DataFrame,
    pitcher_appearances: pd.DataFrame,
    batter_appearances: pd.DataFrame,
    betting_lines: pd.DataFrame,
    pitches_dir: Path,
    bullpen_window_days: int,
    max_lineup_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    cache_dir: Path | None,
    edge_thresholds: list[float],
    stake: float,
    n_resamples: int,
    seed: int | None,
) -> pd.DataFrame:
    """One row per (method, edge_threshold) with that method's
    betting-simulation summary on this fold's test season, at that
    threshold.

    Model inference (GamePredictor forward passes, the LR baseline fit, the
    stacking ensemble fit) happens exactly once per fold regardless of how
    many `edge_thresholds` are swept -- only which bets clear the edge bar
    depends on the threshold, so re-running inference per threshold would be
    pure waste; simulate_flat_stake_betting's edge/profit/CLV computation is
    cheap enough to just repeat for every threshold value.
    """
    system, continuous_stats, rest_day_stats = load_trained_system(checkpoint_path)
    system.to(device)

    games_with_features, _ = build_baseline_features(games)
    train_games = games_with_features[games_with_features["season"].between(*fold.train_seasons)]
    val_games = games_with_features[games_with_features["season"].between(*fold.val_seasons)].sort_values(
        "game_date"
    ).reset_index(drop=True)
    lr_baseline = fit_logistic_regression_baseline(pd.concat([train_games, val_games]))

    test_games = games_with_features[games_with_features["season"].between(*fold.test_seasons)].sort_values(
        "game_date"
    ).reset_index(drop=True)

    full_pitches = read_partitioned(pitches_dir)
    pitches = full_pitches[full_pitches["is_valid"]].reset_index(drop=True)
    inference_args = (
        pitches, pitcher_appearances, batter_appearances, continuous_stats, rest_day_stats,
        bullpen_window_days, max_lineup_size, batch_size, num_workers, device, cache_dir,
    )

    # Stacking is fit on the fold's validation-season predictions -- genuinely
    # out-of-sample for GamePredictor (never used for its gradient updates),
    # same convention as backtest.py's generate_predictions for the fixed split.
    gp_val_probs = run_model_inference(system, val_games, *inference_args)
    lr_val_probs = lr_baseline.predict_proba(val_games[FEATURE_COLUMNS].to_numpy())[:, 1]
    stacking_model = fit_stacking_ensemble(gp_val_probs, lr_val_probs, val_games["home_win"].to_numpy())

    gp_test_probs = run_model_inference(system, test_games, *inference_args)
    lr_test_probs = lr_baseline.predict_proba(test_games[FEATURE_COLUMNS].to_numpy())[:, 1]
    stacked_test_probs = apply_stacking_ensemble(stacking_model, gp_test_probs, lr_test_probs)

    lines = test_games[["game_pk", "home_win"]].merge(betting_lines[BETTING_LINE_COLUMNS], on="game_pk", how="left")
    has_line = lines["home_ml_open"].notna().to_numpy()
    matched_lines = lines.loc[has_line].reset_index(drop=True)
    logger.info("%s: %d/%d test games matched to betting_lines", fold, len(matched_lines), len(test_games))

    rows = []
    for method, probs in [
        (GAME_PREDICTOR_METHOD, gp_test_probs),
        (LOGISTIC_REGRESSION_METHOD, lr_test_probs),
        (ENSEMBLE_METHOD, stacked_test_probs),
    ]:
        for edge_threshold in edge_thresholds:
            result = evaluate_betting_strategy(
                matched_lines, probs[has_line], edge_threshold=edge_threshold, stake=stake,
                n_resamples=n_resamples, seed=seed,
            )
            roi_lo, roi_hi = result["roi_ci95"]
            clv_lo, clv_hi = result["clv_ci95"]
            rows.append(
                {
                    "fold": fold.fold_index,
                    "test_seasons": f"{fold.test_seasons[0]}-{fold.test_seasons[1]}",
                    "method": method,
                    "edge_threshold": edge_threshold,
                    "n_candidates": result["n_candidates"],
                    "n_bets": result["n_bets"],
                    "total_staked": result["total_staked"],
                    "total_profit": result["total_profit"],
                    "roi": result["roi"],
                    "roi_ci_lo": roi_lo,
                    "roi_ci_hi": roi_hi,
                    "roi_significant_positive": roi_lo > 0,
                    "roi_significant_negative": roi_hi < 0,
                    "mean_clv": result["mean_clv"],
                    "clv_ci_lo": clv_lo,
                    "clv_ci_hi": clv_hi,
                    "clv_significant_positive": clv_lo > 0,
                    "clv_significant_negative": clv_hi < 0,
                }
            )
    return pd.DataFrame(rows)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate flat-stake moneyline betting for GamePredictor/LR baseline/stacked ensemble "
        "across all walk-forward folds, using each fold's already-trained checkpoint."
    )
    parser.add_argument("--season-range-start", type=int, default=DEFAULT_SEASON_RANGE[0])
    parser.add_argument("--season-range-end", type=int, default=DEFAULT_SEASON_RANGE[1])
    parser.add_argument("--train-years", type=int, default=DEFAULT_TRAIN_YEARS)
    parser.add_argument("--val-years", type=int, default=DEFAULT_VAL_YEARS)
    parser.add_argument("--test-years", type=int, default=DEFAULT_TEST_YEARS)

    parser.add_argument("--pitches-dir", type=Path, default=PROCESSED_DATA_DIR / "pitches")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--games-dir", type=Path, default=GAMES_DIR)
    parser.add_argument("--pitcher-appearances-dir", type=Path, default=PITCHER_APPEARANCES_DIR)
    parser.add_argument("--batter-appearances-dir", type=Path, default=BATTER_APPEARANCES_DIR)
    parser.add_argument("--betting-lines-dir", type=Path, default=BETTING_LINES_DIR)
    parser.add_argument("--bullpen-window-days", type=int, default=DEFAULT_BULLPEN_WINDOW_DAYS)
    parser.add_argument("--max-lineup-size", type=int, default=DEFAULT_MAX_LINEUP_SIZE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_SEQUENCE_CACHE_DIR))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints") / "walk_forward")

    parser.add_argument(
        "--edge-thresholds",
        type=float,
        nargs="+",
        default=[DEFAULT_EDGE_THRESHOLD],
        help="One or more edge thresholds to sweep (e.g. --edge-thresholds 0.01 0.02 0.05). Model inference "
        "runs once per fold regardless of how many thresholds are given -- only the bet-selection/simulation "
        "step repeats per threshold.",
    )
    parser.add_argument("--stake", type=float, default=DEFAULT_STAKE)
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports") / "betting_sim")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    folds = generate_folds(
        (args.season_range_start, args.season_range_end), args.train_years, args.val_years, args.test_years
    )
    betting_lines = read_partitioned(args.betting_lines_dir)
    logger.info("Loaded %d betting-line rows from %s", len(betting_lines), args.betting_lines_dir)

    fold_frames = []
    for fold in folds:
        checkpoint_path = args.checkpoint_dir / f"fold_{fold.fold_index:02d}_test{fold.test_seasons[0]}.pt"
        if not checkpoint_path.exists():
            logger.warning("%s: no checkpoint at %s, skipping", fold, checkpoint_path)
            continue

        games, pitcher_appearances, batter_appearances = _load_fold_games(
            fold, args.season_range_start, args.raw_dir, args.games_dir,
            args.pitcher_appearances_dir, args.batter_appearances_dir,
        )
        fold_result = evaluate_fold_betting(
            fold, checkpoint_path, games, pitcher_appearances, batter_appearances, betting_lines,
            args.pitches_dir, args.bullpen_window_days, args.max_lineup_size, args.batch_size, args.num_workers,
            device, cache_dir, args.edge_thresholds, args.stake, args.n_resamples, args.seed,
        )
        fold_frames.append(fold_result)
        logger.info("=== %s betting results ===\n%s", fold, fold_result.to_string(index=False))

    if not fold_frames:
        logger.warning("No folds had a trained checkpoint -- nothing to evaluate.")
        return

    results = pd.concat(fold_frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "betting_sim_results.csv"
    results.to_csv(out_path, index=False)

    for (method, edge_threshold), group in results.groupby(["method", "edge_threshold"]):
        total_staked = group["total_staked"].sum()
        pooled_roi = group["total_profit"].sum() / total_staked if total_staked else float("nan")
        logger.info(
            "%s @ edge>%.2f across %d fold(s): %d total bets, pooled ROI=%.4f, "
            "significant positive ROI in %d/%d fold(s), significant positive CLV in %d/%d fold(s)",
            method, edge_threshold, len(group), int(group["n_bets"].sum()), pooled_roi,
            int(group["roi_significant_positive"].sum()), len(group),
            int(group["clv_significant_positive"].sum()), len(group),
        )

    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
