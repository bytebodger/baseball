"""Resumable job (CLAUDE.md's src/resumable_job.py convention): extends a
walk-forward boundary's embedding cache to cover every (player_id,
game_date) pair a full-season simulation-based backtest will need, using
that boundary's own long-history encoder checkpoint (not the project-wide
default one -- mixing encoders would produce embeddings inconsistent with
what the boundary's event model was actually trained against).

Scoped tightly to exactly the pairs season_backtest_fixtures.py's fixtures
need (every starter/bullpen pitcher and every lineup batter, on that game's
own date, across every game in the sample) -- not a broader "every
roster-active date" sweep, since this sample doesn't need one.

Chunked so write_progress() gets called after each chunk, not just once at
the end -- a kill mid-run only loses partial progress within the current
chunk (further bounded by precompute_and_cache_embeddings' own per-entry
incremental writes), and a rerun picks up from the actual current cache
state, not from scratch.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.data.event_embedding_cache import precompute_and_cache_embeddings
from src.data.statcast_common import PROCESSED_DATA_DIR, read_partitioned
from src.device import DEFAULT_DEVICE, resolve_device
from src.evaluation.season_backtest_fixtures import (
    build_fixture,
    load_appearance_tables,
    pitcher_and_batter_query_pairs,
    select_season_games_with_betting_lines,
)
from src.models.long_history_encoder import CareerEncoder, CareerEncoderConfig, ChunkEncoder, ChunkEncoderConfig, LongHistoryEncoder
from src.resumable_job import write_progress

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 500


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend a walk-forward boundary's embedding cache to cover a full-season backtest sample."
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--pitches-dir", type=Path, default=PROCESSED_DATA_DIR / "pitches")
    parser.add_argument("--progress-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)

    logger.info("Selecting season=%d games with real betting lines...", args.season)
    games = select_season_games_with_betting_lines(args.season)
    logger.info("%d games in sample", len(games))

    batter_appearances, pitcher_appearances = load_appearance_tables()
    fixtures = [build_fixture(row, batter_appearances, pitcher_appearances) for _, row in games.iterrows()]
    pitcher_pairs, batter_pairs = pitcher_and_batter_query_pairs(fixtures)
    logger.info("%d distinct pitcher pairs, %d distinct batter pairs needed", len(pitcher_pairs), len(batter_pairs))

    logger.info("Loading encoder checkpoint from %s", args.encoder_checkpoint)
    ckpt = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False)
    chunk_config = ChunkEncoderConfig(**ckpt["chunk_config"])
    career_config = CareerEncoderConfig(**ckpt["career_config"])
    chunk_encoder = ChunkEncoder(chunk_config)
    career_encoder = CareerEncoder(career_config)
    chunk_encoder.load_state_dict(ckpt["chunk_encoder_state_dict"])
    career_encoder.load_state_dict(ckpt["career_encoder_state_dict"])
    encoder = LongHistoryEncoder(chunk_encoder, career_encoder)

    logger.info("Loading full valid pitches (real per-player chunked history)...")
    full_pitches = read_partitioned(args.pitches_dir)
    valid_pitches = full_pitches[full_pitches["is_valid"]].reset_index(drop=True)

    work_items = [("pitcher", p) for p in pitcher_pairs] + [("batter", p) for p in batter_pairs]
    total = len(work_items)
    completed = 0
    write_progress(args.progress_path, total=total, completed=completed, extra={"season": args.season})

    for start in range(0, total, CHUNK_SIZE):
        chunk = work_items[start : start + CHUNK_SIZE]
        chunk_pitcher = [p for perspective, p in chunk if perspective == "pitcher"]
        chunk_batter = [p for perspective, p in chunk if perspective == "batter"]
        queries_by_perspective = {}
        perspectives = []
        if chunk_pitcher:
            queries_by_perspective["pitcher"] = chunk_pitcher
            perspectives.append("pitcher")
        if chunk_batter:
            queries_by_perspective["batter"] = chunk_batter
            perspectives.append("batter")

        precompute_and_cache_embeddings(
            valid_pitches, encoder, args.embedding_cache_dir,
            max_chunks=career_config.max_chunks, max_pitch_len=chunk_config.max_seq_len,
            device=device, batch_size=args.batch_size,
            perspectives=tuple(perspectives), queries_by_perspective=queries_by_perspective,
        )
        completed += len(chunk)
        write_progress(args.progress_path, total=total, completed=completed, extra={"season": args.season})
        logger.info("Progress: %d/%d pairs processed", completed, total)

    logger.info("Done. %d/%d pairs processed.", completed, total)


if __name__ == "__main__":
    main()
