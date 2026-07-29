"""Resumable job (CLAUDE.md's src/resumable_job.py convention): the actual
full-game-simulation leg of the Phase-11-scoped walk-forward backtest
against Vegas for a given boundary. For every real held-out season game
with real betting-line coverage (season_backtest_fixtures.py), simulates
1,000 replays (game_engine.simulate_games_batch, real starters/lineups/
bullpens, one shared GameEngineContext reused across every game -- the
existing low-scoring-game probe's own convention) and records:

- model_home_prob: the model's implied P(home wins), the simulated home
  win rate over the 1,000 replays.
- sim_mean_total_runs: mean simulated total runs for THIS game specifically
  -- what determines its scoring-environment tier (see
  season_backtest_metrics.py's TIER_BOUNDS; fixed thresholds, locked in
  before any result is computed, not adjusted post-hoc based on how results
  look).

One JSONL line per completed game, append-only, written immediately after
that game's own 1,000 sims finish -- a kill/rerun only ever loses whatever
game was mid-simulation, and picks the remaining games up on rerun (see
already_done()).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.evaluation.season_backtest_fixtures import (
    build_fixture,
    load_appearance_tables,
    select_season_games_with_betting_lines,
)
from src.resumable_job import write_progress
from src.simulation.game_engine import build_game_engine_context, simulate_games_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SIM_SEED_BASE = 40000000


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-season simulation-based backtest for one walk-forward boundary.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--n-sims-per-game", type=int, default=1000)
    parser.add_argument("--event-model-checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--contact-quality-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--event-model-season-start", type=int, required=True,
        help="Must match the event-model checkpoint's own train_season_start -- passed through to "
        "build_game_engine_context's event_model_season_range to correctly rebuild the park-factor vocab.",
    )
    parser.add_argument("--event-model-season-end", type=int, required=True, help="Must match the checkpoint's own val_seasons[-1].")
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--progress-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--xbh-rate-checkpoint", type=Path, default=None,
        help="Path to a boundary's leak-safe XBH-rate history (src/data/xbh_rate_history.py) -- required "
        "if --xbh-calibration-gain is nonzero. Must be built from strictly train+val seasons only (never "
        "the season being backtested), same convention as --contact-quality-checkpoint.",
    )
    parser.add_argument(
        "--xbh-calibration-gain", type=float, default=0.0,
        help="Simulation-layer XBH-probability calibration correction (src/simulation/xbh_calibration.py) "
        "-- 0.0 (default) is off, preserving prior behavior exactly. 1.0 fully shifts the model's predicted "
        "XBH share toward the real, leak-safe rolling rate.",
    )
    return parser.parse_args(argv)


def already_done(results_path: Path) -> set[int]:
    if not results_path.exists():
        return set()
    done = set()
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["game_pk"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.xbh_calibration_gain != 0.0 and args.xbh_rate_checkpoint is None:
        raise SystemExit("--xbh-calibration-gain is nonzero but --xbh-rate-checkpoint wasn't given.")

    logger.info("Selecting season=%d games with real betting lines...", args.season)
    games = select_season_games_with_betting_lines(args.season)
    logger.info("%d games in sample", len(games))

    skip = already_done(args.results_path)
    remaining_games = games[~games["game_pk"].isin(skip)]
    write_progress(args.progress_path, total=len(games), completed=len(skip), extra={"season": args.season})
    if skip:
        logger.info("Resuming: %d/%d games already simulated.", len(skip), len(games))

    logger.info(
        "Building game engine context (boundary-specific checkpoints, xbh_calibration_gain=%s)...",
        args.xbh_calibration_gain,
    )
    context_kwargs = dict(
        event_model_checkpoint=args.event_model_checkpoint,
        embedding_cache_dir=args.embedding_cache_dir,
        contact_quality_checkpoint=args.contact_quality_checkpoint,
        event_model_season_range=(args.event_model_season_start, args.event_model_season_end),
        device=args.device,
        xbh_calibration_gain=args.xbh_calibration_gain,
    )
    if args.xbh_rate_checkpoint is not None:
        context_kwargs["xbh_rate_checkpoint"] = args.xbh_rate_checkpoint
    context = build_game_engine_context(**context_kwargs)

    batter_appearances, pitcher_appearances = load_appearance_tables()

    completed = len(skip)
    for _, row in remaining_games.iterrows():
        game_pk = int(row["game_pk"])
        fixture = build_fixture(row, batter_appearances, pitcher_appearances)
        seed = SIM_SEED_BASE + game_pk
        results = simulate_games_batch(
            count=args.n_sims_per_game,
            home_starter=fixture["home_starter"], away_starter=fixture["away_starter"],
            home_lineup=fixture["home_lineup"], away_lineup=fixture["away_lineup"],
            home_bullpen=fixture["home_bullpen"], away_bullpen=fixture["away_bullpen"],
            park_id=fixture["home_team"], game_date=fixture["game_date"],
            context=context, rng=np.random.default_rng(seed),
        )
        home_wins = sum(1 for r in results if r.winner == "home")
        total_runs = np.array([r.home_score + r.away_score for r in results])
        record = {
            "game_pk": game_pk,
            "game_date": fixture["game_date"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "actual_home_win": fixture["actual_home_win"],
            "model_home_prob": home_wins / args.n_sims_per_game,
            "sim_mean_total_runs": float(total_runs.mean()),
        }
        with open(args.results_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        completed += 1
        write_progress(args.progress_path, total=len(games), completed=completed, extra={"season": args.season})
        logger.info(
            "[%d/%d] game_pk=%d %s@%s model_home_prob=%.3f sim_mean_total=%.2f",
            completed, len(games), game_pk, fixture["away_team"], fixture["home_team"],
            record["model_home_prob"], record["sim_mean_total_runs"],
        )

    logger.info("Done. %d/%d games simulated.", completed, len(games))


if __name__ == "__main__":
    main()
