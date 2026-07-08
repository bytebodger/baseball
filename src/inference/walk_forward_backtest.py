"""Walk-forward cross-validation of GamePredictor against the ERA/wRC+
logistic regression baseline.

Every other script in this codebase evaluates against ONE fixed split
(train 2015-2022, validate 2023, test 2024-2025 -- see statcast_common.py).
That's a single data point: if GamePredictor's edge over the baseline is
real, it should hold up across many independent train/test splits, not just
that one. This script slides a 5-year train / 1-year validation / 1-year
test window forward one year at a time across the full season range
(2015-2026 by default), and for each fold:

1. Trains a fresh GamePredictor from the Phase 5 pretrained encoder, using
   the same two-stage fine-tuning pipeline as train_game_predictor.py, just
   with that fold's own train/val seasons instead of the fixed global split.
2. Backtests it against the ERA/wRC+ logistic regression baseline (fit on
   that same fold's train+val seasons, for a fair comparison -- both models
   see the same years of history), the always-home floor, and the Vegas
   closing line (see src/data/build_betting_lines.py) on that fold's test
   season. The Vegas line is a de-vigged implied win probability from
   home_ml_close/away_ml_close -- it's only available for test games that
   have a matched row in the betting_lines table, so it's scored on
   whatever subset of the fold's test games that turns out to be, not the
   full test set.
3. Runs two paired bootstrap comparisons on that fold's test season, the
   same style as bootstrap_compare.py: GamePredictor vs the LR baseline (the
   full test set), and GamePredictor vs the Vegas closing line (restricted
   to the subset with a matched betting line).

Aggregating across folds is the point: a real edge shows up as
consistently favorable direction across folds; an illusory one shows up as
significant in one or two folds and a wash (or reversed) in the rest.

This is expensive -- each fold repeats a full training run (tens of
minutes) plus a full backtest+bootstrap (a few minutes) -- so main() logs
progress per fold and writes results incrementally, in case a long run
needs to be interrupted and inspected partway through.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.data.build_betting_lines import no_vig_home_win_prob
from src.data.game_dataset import (
    BATTER_APPEARANCES_DIR,
    DEFAULT_BULLPEN_WINDOW_DAYS,
    DEFAULT_MAX_LINEUP_SIZE,
    GAMES_DIR,
    PITCHER_APPEARANCES_DIR,
    GameOutcomeDataset,
    ensure_game_tables_built,
)
from src.data.statcast_common import PROCESSED_DATA_DIR, RAW_DATA_DIR, read_partitioned
from src.device import DEFAULT_DEVICE, resolve_device
from src.inference.backtest import (
    FEATURE_COLUMNS,
    build_baseline_features,
    fit_logistic_regression_baseline,
    load_trained_system,
    run_model_inference,
    summarize,
)
from src.inference.bootstrap_compare import bootstrap_compare, summarize_bootstrap
from src.models.game_predictor import GamePredictor
from src.models.set_pooling import DEFAULT_CONFIG_PATH as SET_POOLING_CONFIG_PATH
from src.models.set_pooling import PlayerSetPooler, PlayerSetPoolerConfig
from src.training.train_game_predictor import (
    DEFAULT_ENCODER_CHECKPOINT_PATH,
    DEFAULT_SEQUENCE_CACHE_DIR,
    DEFAULT_TRAINING_CONFIG_PATH,
    EARLY_STOPPING_PATIENCE,
    GameBatchCollator,
    GamePredictionSystem,
    GamePredictorTrainingConfig,
    _compute_rest_day_stats,
    _encoder_trainable_this_epoch,
    load_pretrained_encoder,
    run_epoch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SEASON_RANGE = (2015, 2026)
DEFAULT_TRAIN_YEARS = 5
DEFAULT_VAL_YEARS = 1
DEFAULT_TEST_YEARS = 1
GAME_PREDICTOR_METHOD = "GamePredictor"
LOGISTIC_REGRESSION_METHOD = "Logistic regression (ERA/wRC+ proxy)"
ALWAYS_HOME_METHOD = "Always predict home win"
VEGAS_METHOD = "Vegas closing line"
BETTING_LINES_DIR = PROCESSED_DATA_DIR / "betting_lines"

GP_VS_LR_COMPARISON = "GamePredictor_vs_LR"
GP_VS_VEGAS_COMPARISON = "GamePredictor_vs_Vegas"


@dataclass
class Fold:
    fold_index: int
    train_seasons: tuple[int, int]
    val_seasons: tuple[int, int]
    test_seasons: tuple[int, int]

    def __str__(self) -> str:
        return (
            f"fold {self.fold_index}: train {self.train_seasons[0]}-{self.train_seasons[1]}, "
            f"val {self.val_seasons[0]}-{self.val_seasons[1]}, test {self.test_seasons[0]}-{self.test_seasons[1]}"
        )


def generate_folds(
    season_range: tuple[int, int] = DEFAULT_SEASON_RANGE,
    train_years: int = DEFAULT_TRAIN_YEARS,
    val_years: int = DEFAULT_VAL_YEARS,
    test_years: int = DEFAULT_TEST_YEARS,
) -> list[Fold]:
    """One fold per year the train+val+test window fits inside season_range,
    sliding forward one year each time."""
    start, end = season_range
    folds = []
    train_start = start
    fold_index = 0
    while True:
        train_end = train_start + train_years - 1
        val_start = train_end + 1
        val_end = val_start + val_years - 1
        test_start = val_end + 1
        test_end = test_start + test_years - 1
        if test_end > end:
            break
        folds.append(Fold(fold_index, (train_start, train_end), (val_start, val_end), (test_start, test_end)))
        train_start += 1
        fold_index += 1
    return folds


def _load_fold_games(
    fold: Fold,
    season_range_start: int,
    raw_dir: Path,
    games_dir: Path,
    pitcher_appearances_dir: Path,
    batter_appearances_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (all_games_from_season_range_start_through_this_folds_test_end,
    pitcher_appearances, batter_appearances). Games span from the overall
    season range's own start (not this fold's own train start) through the
    fold's test end, so the ERA/wRC+ proxy features (build_baseline_features)
    get as much real expanding history as actually exists -- same "use all
    available real history for feature engineering" spirit as backtest.py,
    even though the *model* only trains on this fold's own 5-year window."""
    all_seasons = list(range(season_range_start, fold.test_seasons[1] + 1))
    ensure_game_tables_built(all_seasons, raw_dir, games_dir, pitcher_appearances_dir, batter_appearances_dir)
    games = read_partitioned(games_dir).sort_values("game_date").reset_index(drop=True)
    pitcher_appearances = read_partitioned(pitcher_appearances_dir)
    batter_appearances = read_partitioned(batter_appearances_dir)
    return games, pitcher_appearances, batter_appearances


def train_fold(
    fold: Fold,
    encoder_checkpoint: Path,
    training_config: GamePredictorTrainingConfig,
    pooler_config_path: Path,
    games: pd.DataFrame,
    pitcher_appearances: pd.DataFrame,
    batter_appearances: pd.DataFrame,
    pitches_dir: Path,
    bullpen_window_days: int,
    max_lineup_size: int,
    batch_size: int,
    epochs: int,
    patience: int,
    cache_dir: Path | None,
    num_workers: int,
    device: torch.device,
    checkpoint_path: Path,
) -> dict:
    """Trains one fold's GamePredictor from the Phase 5 pretrained encoder,
    the same two-stage pipeline train_game_predictor.py uses, just scoped to
    this fold's own train/val seasons. Saves the best checkpoint to
    `checkpoint_path` using the exact schema train_game_predictor.py itself
    writes, so backtest.py's load_trained_system reads it back unmodified.
    Returns the best epoch's val metrics.
    """
    player_encoder, continuous_stats = load_pretrained_encoder(encoder_checkpoint)

    train_games = games[games["season"].between(*fold.train_seasons)].reset_index(drop=True)
    val_games = games[games["season"].between(*fold.val_seasons)].reset_index(drop=True)

    full_pitches = read_partitioned(pitches_dir)
    pitches = full_pitches[
        full_pitches["season"].between(fold.train_seasons[0], fold.val_seasons[1]) & full_pitches["is_valid"]
    ].reset_index(drop=True)

    train_dataset = GameOutcomeDataset(
        pitches, train_games, pitcher_appearances, batter_appearances,
        player_encoder.config.max_seq_len, bullpen_window_days, max_lineup_size, continuous_stats, cache_dir,
    )
    val_dataset = GameOutcomeDataset(
        pitches, val_games, pitcher_appearances, batter_appearances,
        player_encoder.config.max_seq_len, bullpen_window_days, max_lineup_size, continuous_stats, cache_dir,
    )

    if cache_dir is not None:
        warm_start = time.time()
        for name, ds in [("train", train_dataset), ("val", val_dataset)]:
            pitcher_new, batter_new = ds.warm_cache()
            logger.info(
                "%s %s: computed %d new pitcher + %d new batter sequences", fold, name, pitcher_new, batter_new
            )
        logger.info("%s: cache warm took %.1fs", fold, time.time() - warm_start)

    rest_day_mean, rest_day_std = _compute_rest_day_stats(train_games)
    collate_fn = GameBatchCollator(rest_day_mean, rest_day_std)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers
    )

    loaded_pooling_config = PlayerSetPoolerConfig.from_yaml(pooler_config_path)
    pooler_config = PlayerSetPoolerConfig(
        embed_dim=player_encoder.config.hidden_size,
        num_heads=loaded_pooling_config.num_heads,
        dropout=loaded_pooling_config.dropout,
    )
    bullpen_pooler = PlayerSetPooler(pooler_config)
    lineup_pooler = PlayerSetPooler(pooler_config)

    predictor_config = training_config.to_game_predictor_config()
    game_predictor = GamePredictor(player_encoder, predictor_config)
    system = GamePredictionSystem(game_predictor, bullpen_pooler, lineup_pooler).to(device)

    encoder_param_ids = {id(p) for p in player_encoder.parameters()}
    other_params = [p for p in system.parameters() if id(p) not in encoder_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(player_encoder.parameters()), "lr": training_config.encoder_lr},
            {"params": other_params, "lr": training_config.predictor_lr},
        ]
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)

    best_val_loss = float("inf")
    best_val_metrics: dict = {}
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        encoder_trainable = _encoder_trainable_this_epoch(epoch, training_config)
        train_metrics = run_epoch(
            system, train_loader, device, optimizer=optimizer, scaler=scaler, use_amp=use_amp,
            encoder_trainable=encoder_trainable,
        )
        val_metrics = run_epoch(system, val_loader, device, use_amp=use_amp)
        logger.info(
            "%s epoch %d/%d (encoder %s) - train_loss=%.4f val_loss=%.4f",
            fold, epoch, epochs, "trainable" if encoder_trainable else "frozen",
            train_metrics["loss"], val_metrics["loss"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "game_predictor_state_dict": game_predictor.state_dict(),
                    "bullpen_pooler_state_dict": bullpen_pooler.state_dict(),
                    "lineup_pooler_state_dict": lineup_pooler.state_dict(),
                    "encoder_config": asdict(player_encoder.config),
                    "predictor_config": asdict(predictor_config),
                    "pooler_config": asdict(pooler_config),
                    "continuous_stats": continuous_stats,
                    "rest_day_stats": (rest_day_mean, rest_day_std),
                    "epoch": epoch,
                    "fold": fold.fold_index,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("%s: early stopping at epoch %d", fold, epoch)
                break

    logger.info("%s: done training, best val_loss=%.4f", fold, best_val_loss)
    return best_val_metrics


def backtest_fold(
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
    n_resamples: int,
    seed: int | None,
) -> dict:
    """Backtests one fold's trained checkpoint against the ERA/wRC+ logistic
    regression (fit on this same fold's train+val seasons -- the same years
    of history GamePredictor trained on, for a fair comparison), the
    always-home floor, and the Vegas closing line, on this fold's test
    season. Also runs two paired bootstrap comparisons -- GamePredictor vs
    the LR baseline (on the full test set, as before) and GamePredictor vs
    the Vegas closing line (restricted to whichever test games actually have
    a matched betting line -- see betting_lines, not every game does).
    Returns a dict with this fold's summary table and both bootstrap
    summaries.
    """
    system, continuous_stats, rest_day_stats = load_trained_system(checkpoint_path)
    system.to(device)

    games_with_features, league_avg_runs = build_baseline_features(games)
    train_games_for_lr = games_with_features[games_with_features["season"].between(*fold.train_seasons)]
    # LR baseline is fit on this fold's train+val (same as GamePredictor's own
    # training scope), matching backtest.py's train+val convention for the
    # global split, just re-scoped per fold.
    lr_fit_games = pd.concat(
        [train_games_for_lr, games_with_features[games_with_features["season"].between(*fold.val_seasons)]]
    )
    lr_baseline = fit_logistic_regression_baseline(lr_fit_games)

    test_games = games_with_features[games_with_features["season"].between(*fold.test_seasons)].sort_values(
        "game_date"
    ).reset_index(drop=True)

    full_pitches = read_partitioned(pitches_dir)
    pitches = full_pitches[full_pitches["is_valid"]].reset_index(drop=True)

    gp_probs = run_model_inference(
        system, test_games, pitches, pitcher_appearances, batter_appearances,
        continuous_stats, rest_day_stats, bullpen_window_days, max_lineup_size, batch_size, num_workers, device,
        cache_dir,
    )
    lr_probs = lr_baseline.predict_proba(test_games[FEATURE_COLUMNS].to_numpy())[:, 1]
    always_home_probs = np.ones(len(test_games))
    y_true = test_games["home_win"].to_numpy().astype(float)

    test_lines = test_games[["game_pk"]].merge(
        betting_lines[["game_pk", "home_ml_close", "away_ml_close"]], on="game_pk", how="left"
    )
    has_line = test_lines["home_ml_close"].notna().to_numpy()
    vegas_probs = no_vig_home_win_prob(
        test_lines.loc[has_line, "home_ml_close"], test_lines.loc[has_line, "away_ml_close"]
    )
    logger.info(
        "%s: matched Vegas closing lines for %d/%d test games", fold, int(has_line.sum()), len(test_games)
    )

    results = {
        GAME_PREDICTOR_METHOD: (y_true, gp_probs),
        LOGISTIC_REGRESSION_METHOD: (y_true, lr_probs),
        ALWAYS_HOME_METHOD: (y_true, always_home_probs),
        VEGAS_METHOD: (y_true[has_line], vegas_probs),
    }
    summary = summarize(results)
    summary.insert(0, "fold", fold.fold_index)
    summary.insert(1, "test_seasons", f"{fold.test_seasons[0]}-{fold.test_seasons[1]}")

    bootstrap_summaries = []
    for comparison, y, probs_a, probs_b in [
        (GP_VS_LR_COMPARISON, y_true, gp_probs, lr_probs),
        (GP_VS_VEGAS_COMPARISON, y_true[has_line], gp_probs[has_line], vegas_probs),
    ]:
        bootstrap_results = bootstrap_compare(y, probs_a, probs_b, n_resamples=n_resamples, seed=seed)
        bootstrap_summary = summarize_bootstrap(bootstrap_results)
        bootstrap_summary["comparison"] = comparison
        bootstrap_summary["fold"] = fold.fold_index
        bootstrap_summary["test_seasons"] = f"{fold.test_seasons[0]}-{fold.test_seasons[1]}"
        bootstrap_summaries.append(bootstrap_summary)

    return {"summary": summary, "bootstrap_summaries": bootstrap_summaries}


def aggregate_fold_bootstraps(fold_bootstrap_summaries: list[dict]) -> pd.DataFrame:
    """One row per (fold, comparison), plus columns flagging whether that
    row's 95% CI excludes zero in GamePredictor's favor / the other method's
    favor / neither (straddles zero -- inconclusive)."""
    rows = []
    for s in fold_bootstrap_summaries:
        acc_lo, acc_hi = s["accuracy_diff_ci95"]
        brier_lo, brier_hi = s["brier_diff_ci95"]
        rows.append(
            {
                "fold": s["fold"],
                "test_seasons": s["test_seasons"],
                "comparison": s["comparison"],
                "accuracy_diff_mean": s["accuracy_diff_mean"],
                "accuracy_diff_ci_lo": acc_lo,
                "accuracy_diff_ci_hi": acc_hi,
                "accuracy_significant_for_gamepredictor": acc_lo > 0,
                "accuracy_significant_for_baseline": acc_hi < 0,
                "brier_diff_mean": s["brier_diff_mean"],
                "brier_diff_ci_lo": brier_lo,
                "brier_diff_ci_hi": brier_hi,
                # lower Brier is better, so GamePredictor is favored when the
                # whole CI is below zero.
                "brier_significant_for_gamepredictor": brier_hi < 0,
                "brier_significant_for_baseline": brier_lo > 0,
            }
        )
    return pd.DataFrame(rows)


COMPARISON_LABELS = {
    GP_VS_LR_COMPARISON: "GamePredictor vs ERA/wRC+ logistic regression",
    GP_VS_VEGAS_COMPARISON: "GamePredictor vs Vegas closing line",
}


def plot_fold_effects(aggregate: pd.DataFrame, comparison: str, output_path: Path) -> None:
    """Plots one comparison's fold effects (`aggregate` should already be
    filtered to a single `comparison` value -- see main(), which plots each
    comparison to its own file)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    label = COMPARISON_LABELS.get(comparison, comparison)

    for ax, prefix, title in [
        (axes[0], "accuracy_diff", f"Accuracy difference ({label.replace(' vs ', ' - ')})"),
        (axes[1], "brier_diff", f"Brier score difference ({label.replace(' vs ', ' - ')})"),
    ]:
        x = aggregate["test_seasons"]
        mean = aggregate[f"{prefix}_mean"]
        lo = aggregate[f"{prefix}_ci_lo"]
        hi = aggregate[f"{prefix}_ci_hi"]
        # The percentile CI can (rarely, especially for a skewed distribution
        # or a small n_resamples) not bracket the arithmetic mean exactly --
        # clip rather than let matplotlib's errorbar reject a negative span.
        yerr = np.stack([(mean - lo).clip(lower=0), (hi - mean).clip(lower=0)])

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.5)
        ax.errorbar(x, mean, yerr=yerr, fmt="o", color="steelblue", capsize=4, markersize=6)
        ax.set_title(title)
        ax.set_xlabel("Test season")
        ax.set_ylabel("Difference (95% CI)")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(f"Walk-forward fold effects: {label}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward cross-validated backtest of GamePredictor against the ERA/wRC+ LR baseline."
    )
    parser.add_argument("--season-range-start", type=int, default=DEFAULT_SEASON_RANGE[0])
    parser.add_argument("--season-range-end", type=int, default=DEFAULT_SEASON_RANGE[1])
    parser.add_argument("--train-years", type=int, default=DEFAULT_TRAIN_YEARS)
    parser.add_argument("--val-years", type=int, default=DEFAULT_VAL_YEARS)
    parser.add_argument("--test-years", type=int, default=DEFAULT_TEST_YEARS)

    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG_PATH)
    parser.add_argument("--set-pooling-config", type=Path, default=SET_POOLING_CONFIG_PATH)
    parser.add_argument("--encoder-checkpoint", type=Path, default=DEFAULT_ENCODER_CHECKPOINT_PATH)
    parser.add_argument("--pitches-dir", type=Path, default=PROCESSED_DATA_DIR / "pitches")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--games-dir", type=Path, default=GAMES_DIR)
    parser.add_argument("--pitcher-appearances-dir", type=Path, default=PITCHER_APPEARANCES_DIR)
    parser.add_argument("--batter-appearances-dir", type=Path, default=BATTER_APPEARANCES_DIR)
    parser.add_argument("--betting-lines-dir", type=Path, default=BETTING_LINES_DIR)
    parser.add_argument("--bullpen-window-days", type=int, default=DEFAULT_BULLPEN_WINDOW_DAYS)
    parser.add_argument("--max-lineup-size", type=int, default=DEFAULT_MAX_LINEUP_SIZE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Defaults to cuda -- this project trains on GPU. Pass --device cpu to explicitly opt into a "
        "(much slower) CPU run instead of silently falling back to one.",
    )
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_SEQUENCE_CACHE_DIR))
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints") / "walk_forward")
    parser.add_argument("--output-dir", type=Path, default=Path("reports") / "walk_forward")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip folds that already have a completed row in fold_bootstrap_summary.csv (in --output-dir) "
        "instead of retraining/re-backtesting them -- for picking a long run back up after an interruption.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain a fold's GamePredictor even if --checkpoint-dir already has a saved checkpoint for it. "
        "By default an existing checkpoint is reused as-is and only the backtest/bootstrap step is rerun -- "
        "useful for re-running backtest_fold (e.g. after adding a new baseline method) without repeating "
        "the expensive training step for folds that already have a trained model.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    training_config = GamePredictorTrainingConfig.from_yaml(args.training_config)
    folds = generate_folds(
        (args.season_range_start, args.season_range_end), args.train_years, args.val_years, args.test_years
    )
    logger.info("Generated %d folds:", len(folds))
    for fold in folds:
        logger.info("  %s", fold)

    betting_lines = read_partitioned(args.betting_lines_dir)
    logger.info("Loaded %d betting-line rows from %s", len(betting_lines), args.betting_lines_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "fold_summaries.csv"
    bootstrap_path = args.output_dir / "fold_bootstrap_summary.csv"

    completed_fold_indices: set[int] = set()
    summary_frames: list[pd.DataFrame] = []
    aggregate_frames: list[pd.DataFrame] = []

    if args.resume and summary_path.exists() and bootstrap_path.exists():
        existing_summary = pd.read_csv(summary_path)
        existing_aggregate = pd.read_csv(bootstrap_path)
        completed_fold_indices = set(existing_aggregate["fold"].tolist())
        summary_frames.append(existing_summary)
        aggregate_frames.append(existing_aggregate)
        logger.info(
            "Resuming: found %d already-completed fold(s) in %s: %s",
            len(completed_fold_indices), args.output_dir, sorted(completed_fold_indices),
        )

    for fold in folds:
        if fold.fold_index in completed_fold_indices:
            logger.info("Skipping %s (already completed, --resume)", fold)
            continue

        fold_start = time.time()
        logger.info("=== Starting %s ===", fold)

        games, pitcher_appearances, batter_appearances = _load_fold_games(
            fold, args.season_range_start, args.raw_dir, args.games_dir,
            args.pitcher_appearances_dir, args.batter_appearances_dir,
        )
        checkpoint_path = args.checkpoint_dir / f"fold_{fold.fold_index:02d}_test{fold.test_seasons[0]}.pt"

        if checkpoint_path.exists() and not args.retrain:
            logger.info("%s: reusing existing checkpoint %s (pass --retrain to force retraining)", fold, checkpoint_path)
        else:
            train_fold(
                fold, args.encoder_checkpoint, training_config, args.set_pooling_config,
                games, pitcher_appearances, batter_appearances, args.pitches_dir,
                args.bullpen_window_days, args.max_lineup_size, args.batch_size, args.epochs, args.patience,
                cache_dir, args.num_workers, device, checkpoint_path,
            )

        fold_result = backtest_fold(
            fold, checkpoint_path, games, pitcher_appearances, batter_appearances, betting_lines, args.pitches_dir,
            args.bullpen_window_days, args.max_lineup_size, args.batch_size, args.num_workers, device,
            cache_dir, args.n_resamples, args.seed,
        )
        summary_frames.append(fold_result["summary"])
        aggregate_frames.append(aggregate_fold_bootstraps(fold_result["bootstrap_summaries"]))

        logger.info(
            "=== %s done in %.1f min ===\n%s", fold, (time.time() - fold_start) / 60,
            fold_result["summary"].to_string(index=False),
        )

        # write incrementally so a long run can be inspected/interrupted partway through
        pd.concat(summary_frames, ignore_index=True).to_csv(summary_path, index=False)
        pd.concat(aggregate_frames, ignore_index=True).sort_values(["comparison", "fold"]).to_csv(
            bootstrap_path, index=False
        )

    aggregate = pd.concat(aggregate_frames, ignore_index=True).sort_values(["comparison", "fold"]).reset_index(drop=True)

    plot_paths = []
    for comparison, group in aggregate.groupby("comparison"):
        group = group.reset_index(drop=True)
        n_folds = len(group)
        acc_sig_gp = int(group["accuracy_significant_for_gamepredictor"].sum())
        acc_sig_base = int(group["accuracy_significant_for_baseline"].sum())
        brier_sig_gp = int(group["brier_significant_for_gamepredictor"].sum())
        brier_sig_base = int(group["brier_significant_for_baseline"].sum())
        label = COMPARISON_LABELS.get(comparison, comparison)

        logger.info(
            "\nWalk-forward summary across %d folds -- %s:\n"
            "  Accuracy: GamePredictor significantly better in %d/%d folds, other method significantly better "
            "in %d/%d, inconclusive in %d/%d. Mean accuracy_diff across folds: %.4f (std %.4f)\n"
            "  Brier:    GamePredictor significantly better in %d/%d folds, other method significantly better "
            "in %d/%d, inconclusive in %d/%d. Mean brier_diff across folds: %.4f (std %.4f)",
            n_folds, label,
            acc_sig_gp, n_folds, acc_sig_base, n_folds, n_folds - acc_sig_gp - acc_sig_base, n_folds,
            group["accuracy_diff_mean"].mean(), group["accuracy_diff_mean"].std(),
            brier_sig_gp, n_folds, brier_sig_base, n_folds, n_folds - brier_sig_gp - brier_sig_base, n_folds,
            group["brier_diff_mean"].mean(), group["brier_diff_mean"].std(),
        )

        plot_path = args.output_dir / f"fold_effects_{comparison.lower()}.png"
        plot_fold_effects(group, comparison, plot_path)
        plot_paths.append(plot_path.name)

    logger.info(
        "Wrote fold_summaries.csv, fold_bootstrap_summary.csv, and %s to %s",
        ", ".join(plot_paths), args.output_dir,
    )


if __name__ == "__main__":
    main()
