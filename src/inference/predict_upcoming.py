"""Predicts win probability for upcoming games using the trained
GamePredictor, the ERA/wRC+ logistic-regression baseline, and the stacked
meta-model ensemble that blends them (see backtest.py) -- loading the
ensemble's already-fitted weights (EnsembleArtifacts, saved by
backtest.py's main()) rather than refitting anything here.

"Upcoming" games are whichever games are in the processed games table for
--season (default: the season right after TEST_SEASON_RANGE, i.e. genuinely
new data no part of training/validation/backtesting has ever touched),
optionally further filtered to --as-of-date onward. This pipeline has no
live schedule/probable-starters feed -- game_dataset.py's own docstring
notes starters and lineups are derived retroactively from a game's own
actual Statcast pitches -- so it can only "predict" games that have already
been played and ingested into the processed pitch data, not literally
future/unscheduled ones. In practice that's still useful: --season defaults
to whatever the next real season is (2026 as of this pipeline's data), which
is real data no part of the trained pipeline has touched, the closest thing
to "upcoming" this pipeline can honestly support.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data.game_dataset import ensure_game_tables_built
from src.data.statcast_common import TEST_SEASON_RANGE, TRAIN_SEASON_RANGE, read_partitioned
from src.device import resolve_device
from src.inference.backtest import (
    DEFAULT_ENSEMBLE_PATH,
    FEATURE_COLUMNS,
    add_common_args,
    apply_stacking_ensemble,
    build_baseline_features,
    load_trained_system,
    run_model_inference,
)
from src.inference.ensemble_artifacts import load_ensemble_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_UPCOMING_SEASON = TEST_SEASON_RANGE[1] + 1


def predict_upcoming(args: argparse.Namespace) -> pd.DataFrame:
    """Returns one row per upcoming game: game_pk, game_date, home_team,
    away_team, and each method's predicted home-win probability, with
    ensemble_win_prob (the stacked meta-model) as the headline column."""
    device = resolve_device(args.device)

    logger.info("Loading trained GamePredictor system from %s", args.game_predictor_checkpoint)
    system, continuous_stats, rest_day_stats = load_trained_system(args.game_predictor_checkpoint)
    system.to(device)

    logger.info("Loading fitted ensemble weights from %s", args.ensemble_path)
    ensemble_artifacts = load_ensemble_artifacts(args.ensemble_path)

    all_seasons = list(range(TRAIN_SEASON_RANGE[0], args.season + 1))
    ensure_game_tables_built(all_seasons, args.raw_dir, args.games_dir, args.pitcher_appearances_dir, args.batter_appearances_dir)
    all_games = read_partitioned(args.games_dir).sort_values("game_date").reset_index(drop=True)
    pitcher_appearances = read_partitioned(args.pitcher_appearances_dir)
    batter_appearances = read_partitioned(args.batter_appearances_dir)

    logger.info("Computing aggregate-stat baseline features (starter ERA proxy, team wRC+ proxy)")
    all_games, _ = build_baseline_features(all_games)

    upcoming_games = all_games[all_games["season"] == args.season].sort_values("game_date").reset_index(drop=True)
    if args.as_of_date:
        upcoming_games = upcoming_games[upcoming_games["game_date"] >= pd.Timestamp(args.as_of_date)].reset_index(drop=True)

    if len(upcoming_games) == 0:
        logger.warning("No games found for season=%d as_of_date=%s", args.season, args.as_of_date)
        return pd.DataFrame(
            columns=["game_pk", "game_date", "home_team", "away_team", "gamepredictor_win_prob", "lr_baseline_win_prob", "ensemble_win_prob"]
        )

    logger.info("Predicting %d upcoming games (season=%d, as_of_date=%s)", len(upcoming_games), args.season, args.as_of_date)

    full_pitches = read_partitioned(args.pitches_dir)
    pitches = full_pitches[full_pitches["is_valid"]].reset_index(drop=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    gp_probs = run_model_inference(
        system, upcoming_games, pitches, pitcher_appearances, batter_appearances,
        continuous_stats, rest_day_stats, args.bullpen_window_days, args.max_lineup_size,
        args.batch_size, args.num_workers, device, cache_dir,
    )
    lr_probs = ensemble_artifacts.lr_baseline.predict_proba(upcoming_games[FEATURE_COLUMNS].to_numpy())[:, 1]
    ensemble_probs = apply_stacking_ensemble(ensemble_artifacts.stacking_model, gp_probs, lr_probs)

    return pd.DataFrame(
        {
            "game_pk": upcoming_games["game_pk"],
            "game_date": upcoming_games["game_date"],
            "home_team": upcoming_games["home_team"],
            "away_team": upcoming_games["away_team"],
            "gamepredictor_win_prob": gp_probs,
            "lr_baseline_win_prob": lr_probs,
            "ensemble_win_prob": ensemble_probs,
            "home_win_prob": ensemble_probs,
            "away_win_prob": 1.0 - ensemble_probs,
        }
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict home-win probability for upcoming games using the stacked GamePredictor+LR ensemble."
    )
    add_common_args(parser)
    parser.add_argument("--season", type=int, default=DEFAULT_UPCOMING_SEASON)
    parser.add_argument(
        "--as-of-date", default=None, help="Only predict games on/after this date (YYYY-MM-DD). Default: the whole season."
    )
    parser.add_argument(
        "--ensemble-path",
        type=Path,
        default=DEFAULT_ENSEMBLE_PATH,
        help="Fitted LR baseline + stacking ensemble weights, saved by backtest.py's main().",
    )
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional path to also save predictions as CSV.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    predictions = predict_upcoming(args)

    if len(predictions) > 0:
        logger.info("Upcoming game predictions:\n%s", predictions.to_string(index=False))
        matchup_lines = "\n".join(
            f"{row.game_date.date()}  {row.away_team} @ {row.home_team}  --  "
            f"{row.away_team} {row.away_win_prob:.1%}  vs  {row.home_team} {row.home_win_prob:.1%}"
            for row in predictions.itertuples()
        )
        logger.info("Matchups (ensemble win probability):\n%s", matchup_lines)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output_csv, index=False)
        logger.info("Wrote predictions to %s", args.output_csv)


if __name__ == "__main__":
    main()
