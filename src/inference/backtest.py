"""Walk-forward backtest of the trained GamePredictor (Phase 9 checkpoint)
against two baselines, over the 2024-2025 held-out seasons (TEST_SEASON_RANGE
-- never touched by training or validation).

For every game in the held-out range, in chronological order, all three
methods predict using only information dated strictly before that game:

- GamePredictor: this is inherent to how its inputs are built --
  GameOutcomeDataset's per-player pitch histories, trailing-window bullpen,
  and starting lineup are already computed "as of" each game's own date (see
  src/data/game_dataset.py), so no extra walk-forward bookkeeping is needed
  here beyond loading it for the held-out seasons.
- Logistic regression on aggregate stats (starter ERA proxy, team wRC+ proxy,
  home/away): its two features per side are *expanding* averages computed
  from every game strictly before the game in question (see
  _add_starter_era_proxy / _add_team_wrc_plus_proxy), which is mathematically
  the same "walk forward one game at a time" computation as an explicit
  per-game loop, just vectorized. The regression itself is fit once on the
  train+val seasons (2015-2023, matching GamePredictor's own split) and then
  applied forward across the held-out seasons -- exactly how GamePredictor
  itself is a fixed, already-trained model being walked forward, not
  refit game by game.
- Always predict home team wins: win_prob = 1.0 for every game, the floor
  any real model needs to beat.

Two notes on the aggregate-stat baseline, since neither stat is available
verbatim from this pipeline's tables:
- "ERA proxy" is a start's team runs-allowed, expanding-averaged over that
  same pitcher's own previous starts -- not a real earned-run breakdown
  (this pipeline has no per-pitcher earned-vs-unearned attribution), same
  "good enough, honestly labeled" spirit as game_dataset.py's own bullpen-
  availability proxy.
- "wRC+ proxy" is a team's runs-scored-per-game, expanding-averaged over its
  own previous games, indexed to 100 = the league's average runs/game across
  the training seasons (not the real wRC+, which needs linear weights per
  plate-appearance event and park factors -- not available here either).

Outputs a summary table (accuracy + Brier score per method) and a calibration
plot, both written to --output-dir.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from src.data.game_dataset import (
    BATTER_APPEARANCES_DIR,
    DEFAULT_BULLPEN_WINDOW_DAYS,
    DEFAULT_MAX_LINEUP_SIZE,
    GAMES_DIR,
    PITCHER_APPEARANCES_DIR,
    GameOutcomeDataset,
    ensure_game_tables_built,
)
from src.data.statcast_common import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TEST_SEASON_RANGE,
    TRAIN_SEASON_RANGE,
    VAL_SEASONS,
    read_partitioned,
)
from src.models.game_predictor import GamePredictor, GamePredictorConfig
from src.models.player_encoder import PlayerEncoder, PlayerEncoderConfig
from src.models.set_pooling import PlayerSetPooler, PlayerSetPoolerConfig
from src.training.train_game_predictor import GameBatchCollator, GamePredictionSystem, _move_batch_to_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_GAME_PREDICTOR_CHECKPOINT = Path("checkpoints") / "game_predictor_best.pt"
FEATURE_COLUMNS = ["home_era_proxy", "away_era_proxy", "home_wrc_plus_proxy", "away_wrc_plus_proxy"]


def _add_starter_era_proxy(games: pd.DataFrame) -> pd.DataFrame:
    """Adds home_era_proxy/away_era_proxy: each starter's own expanding
    average of runs allowed (the opposing team's score) in their previous
    starts, strictly excluding the current game (the `.shift(1)`)."""
    starts = pd.concat(
        [
            games[["game_pk", "game_date", "home_starter_id", "away_score"]]
            .rename(columns={"home_starter_id": "pitcher_id", "away_score": "runs_allowed"})
            .assign(side="home"),
            games[["game_pk", "game_date", "away_starter_id", "home_score"]]
            .rename(columns={"away_starter_id": "pitcher_id", "home_score": "runs_allowed"})
            .assign(side="away"),
        ]
    ).sort_values(["pitcher_id", "game_date"])
    starts["era_proxy"] = starts.groupby("pitcher_id")["runs_allowed"].transform(
        lambda s: s.expanding().mean().shift(1)
    )

    home = starts[starts["side"] == "home"][["game_pk", "era_proxy"]].rename(columns={"era_proxy": "home_era_proxy"})
    away = starts[starts["side"] == "away"][["game_pk", "era_proxy"]].rename(columns={"era_proxy": "away_era_proxy"})
    return games.merge(home, on="game_pk", how="left").merge(away, on="game_pk", how="left")


def _add_team_wrc_plus_proxy(games: pd.DataFrame, league_avg_runs: float) -> pd.DataFrame:
    """Adds home_wrc_plus_proxy/away_wrc_plus_proxy: each team's own
    expanding average runs scored in its previous games (again strictly
    excluding the current game), indexed to 100 = league_avg_runs."""
    team_games = pd.concat(
        [
            games[["game_pk", "game_date", "home_team", "home_score"]]
            .rename(columns={"home_team": "team", "home_score": "runs_scored"})
            .assign(side="home"),
            games[["game_pk", "game_date", "away_team", "away_score"]]
            .rename(columns={"away_team": "team", "away_score": "runs_scored"})
            .assign(side="away"),
        ]
    ).sort_values(["team", "game_date"])
    team_games["wrc_plus_proxy"] = (
        100.0
        * team_games.groupby("team")["runs_scored"].transform(lambda s: s.expanding().mean().shift(1))
        / league_avg_runs
    )

    home = team_games[team_games["side"] == "home"][["game_pk", "wrc_plus_proxy"]].rename(
        columns={"wrc_plus_proxy": "home_wrc_plus_proxy"}
    )
    away = team_games[team_games["side"] == "away"][["game_pk", "wrc_plus_proxy"]].rename(
        columns={"wrc_plus_proxy": "away_wrc_plus_proxy"}
    )
    return games.merge(home, on="game_pk", how="left").merge(away, on="game_pk", how="left")


def build_baseline_features(games: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Adds FEATURE_COLUMNS to `games` (sorted by game_date, spanning train
    through test seasons) and fills each team/pitcher's first-ever
    appearance in the data (which has no prior starts to average) with the
    league-average value -- an average prior is the only leakage-free
    default available before any of that player's own history exists."""
    games = _add_starter_era_proxy(games)
    train_mask = games["season"].between(TRAIN_SEASON_RANGE[0], VAL_SEASONS[-1])
    league_avg_runs = float(
        pd.concat([games.loc[train_mask, "home_score"], games.loc[train_mask, "away_score"]]).mean()
    )
    games = _add_team_wrc_plus_proxy(games, league_avg_runs)

    fill_values = {
        "home_era_proxy": league_avg_runs,
        "away_era_proxy": league_avg_runs,
        "home_wrc_plus_proxy": 100.0,
        "away_wrc_plus_proxy": 100.0,
    }
    games = games.fillna(fill_values)
    return games, league_avg_runs


def fit_logistic_regression_baseline(train_games: pd.DataFrame) -> LogisticRegression:
    model = LogisticRegression()
    model.fit(train_games[FEATURE_COLUMNS].to_numpy(), train_games["home_win"].to_numpy())
    return model


def load_trained_system(checkpoint_path: Path) -> tuple[GamePredictionSystem, dict[str, tuple[float, float]], tuple[float, float]]:
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    encoder = PlayerEncoder(PlayerEncoderConfig(**checkpoint["encoder_config"]))
    predictor_config = GamePredictorConfig(**checkpoint["predictor_config"])
    game_predictor = GamePredictor(encoder, predictor_config)
    game_predictor.load_state_dict(checkpoint["game_predictor_state_dict"])

    pooler_config = PlayerSetPoolerConfig(**checkpoint["pooler_config"])
    bullpen_pooler = PlayerSetPooler(pooler_config)
    bullpen_pooler.load_state_dict(checkpoint["bullpen_pooler_state_dict"])
    lineup_pooler = PlayerSetPooler(pooler_config)
    lineup_pooler.load_state_dict(checkpoint["lineup_pooler_state_dict"])

    system = GamePredictionSystem(game_predictor, bullpen_pooler, lineup_pooler)
    system.eval()
    return system, checkpoint["continuous_stats"], tuple(checkpoint["rest_day_stats"])


@torch.no_grad()
def run_model_inference(
    system: GamePredictionSystem,
    test_games: pd.DataFrame,
    pitches: pd.DataFrame,
    pitcher_appearances: pd.DataFrame,
    batter_appearances: pd.DataFrame,
    continuous_stats: dict[str, tuple[float, float]],
    rest_day_stats: tuple[float, float],
    bullpen_window_days: int,
    max_lineup_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    dataset = GameOutcomeDataset(
        pitches, test_games, pitcher_appearances, batter_appearances,
        system.game_predictor.player_encoder.config.max_seq_len,
        bullpen_window_days, max_lineup_size, continuous_stats,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=GameBatchCollator(*rest_day_stats), num_workers=num_workers,
    )

    win_probs = []
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        output = system(batch)
        win_probs.append(output["win_prob"].cpu())
    return torch.cat(win_probs).numpy()


def summarize(results: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """results: method name -> (y_true, win_prob)."""
    rows = []
    for name, (y_true, win_prob) in results.items():
        predicted_home_win = win_prob >= 0.5
        rows.append(
            {
                "method": name,
                "n_games": len(y_true),
                "accuracy": float((predicted_home_win == (y_true == 1.0)).mean()),
                "brier_score": float(np.mean((win_prob - y_true) ** 2)),
            }
        )
    return pd.DataFrame(rows)


def plot_calibration(results: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path) -> None:
    """results: method name -> (y_true, win_prob)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    for name, (y_true, win_prob) in results.items():
        # "uniform" bins (fixed-width over [0, 1]) rather than "quantile":
        # the always-home baseline's win_prob is a constant 1.0, and
        # quantile binning errors on duplicate bin edges for a constant array.
        fraction_of_positives, mean_predicted = calibration_curve(y_true, win_prob, n_bins=10, strategy="uniform")
        ax.plot(mean_predicted, fraction_of_positives, marker="o", label=name)

    ax.set_xlabel("Mean predicted win probability")
    ax.set_ylabel("Observed home-win frequency")
    ax.set_title(f"Win-probability calibration -- {TEST_SEASON_RANGE[0]}-{TEST_SEASON_RANGE[1]} held-out seasons")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest of GamePredictor against aggregate-stat and always-home baselines."
    )
    parser.add_argument("--game-predictor-checkpoint", type=Path, default=DEFAULT_GAME_PREDICTOR_CHECKPOINT)
    parser.add_argument("--pitches-dir", type=Path, default=PROCESSED_DATA_DIR / "pitches")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--games-dir", type=Path, default=GAMES_DIR)
    parser.add_argument("--pitcher-appearances-dir", type=Path, default=PITCHER_APPEARANCES_DIR)
    parser.add_argument("--batter-appearances-dir", type=Path, default=BATTER_APPEARANCES_DIR)
    parser.add_argument("--bullpen-window-days", type=int, default=DEFAULT_BULLPEN_WINDOW_DAYS)
    parser.add_argument("--max-lineup-size", type=int, default=DEFAULT_MAX_LINEUP_SIZE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("reports") / "backtest")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)

    logger.info("Loading trained GamePredictor system from %s", args.game_predictor_checkpoint)
    system, continuous_stats, rest_day_stats = load_trained_system(args.game_predictor_checkpoint)
    system.to(device)

    all_seasons = list(range(TRAIN_SEASON_RANGE[0], TEST_SEASON_RANGE[1] + 1))
    ensure_game_tables_built(all_seasons, args.raw_dir, args.games_dir, args.pitcher_appearances_dir, args.batter_appearances_dir)
    all_games = read_partitioned(args.games_dir).sort_values("game_date").reset_index(drop=True)
    pitcher_appearances = read_partitioned(args.pitcher_appearances_dir)
    batter_appearances = read_partitioned(args.batter_appearances_dir)

    logger.info("Computing walk-forward aggregate-stat baseline features (starter ERA proxy, team wRC+ proxy)")
    all_games, league_avg_runs = build_baseline_features(all_games)

    train_games = all_games[all_games["season"].between(TRAIN_SEASON_RANGE[0], VAL_SEASONS[-1])]
    logger.info("Fitting logistic-regression baseline on %d train/val games (league avg runs/game=%.2f)", len(train_games), league_avg_runs)
    lr_baseline = fit_logistic_regression_baseline(train_games)

    test_games = all_games[all_games["season"].between(*TEST_SEASON_RANGE)].sort_values("game_date").reset_index(drop=True)
    logger.info("Walking forward through %d held-out games (%d-%d)", len(test_games), *TEST_SEASON_RANGE)

    full_pitches = read_partitioned(args.pitches_dir)
    pitches = full_pitches[full_pitches["is_valid"]].reset_index(drop=True)

    model_win_probs = run_model_inference(
        system, test_games, pitches, pitcher_appearances, batter_appearances,
        continuous_stats, rest_day_stats, args.bullpen_window_days, args.max_lineup_size,
        args.batch_size, args.num_workers, device,
    )
    lr_win_probs = lr_baseline.predict_proba(test_games[FEATURE_COLUMNS].to_numpy())[:, 1]
    always_home_win_probs = np.ones(len(test_games))

    y_true = test_games["home_win"].to_numpy().astype(float)
    results = {
        "GamePredictor": (y_true, model_win_probs),
        "Logistic regression (ERA/wRC+ proxy)": (y_true, lr_win_probs),
        "Always predict home win": (y_true, always_home_win_probs),
    }

    summary = summarize(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "backtest_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Backtest summary (%d-%d, %d games):\n%s", *TEST_SEASON_RANGE, len(test_games), summary.to_string(index=False))

    plot_path = args.output_dir / "calibration_plot.png"
    plot_calibration(results, plot_path)

    logger.info("Wrote summary table to %s and calibration plot to %s", summary_path, plot_path)


if __name__ == "__main__":
    main()
