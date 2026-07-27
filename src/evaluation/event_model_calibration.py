"""Aggregate marginal-probability calibration check for a trained EventModel
checkpoint (CLAUDE.md standing convention): compares the model's average
predicted probability per OUTCOME_VOCAB category, over every real held-out
(validation-split) pitch, against the real observed marginal frequency for
that category. Run this immediately after any event-model retrain, before
any simulation-based validation (paired-pitcher probes, full-game
simulation) -- it's seconds, not minutes, and catches a class of failure
those simulation-based checks can miss entirely: a 2026-07 incident (see
CLAUDE.md) had a training-loss change pass both of this project's
simulation-based probes' *relative* comparisons while being badly
miscalibrated in *absolute* terms (predicted extra-base-hit rate ~3x real,
~59 simulated runs/game vs. a real ~9).

For "predicted probability," this uses the model's own predicted softmax
distribution's marginal, not just the argmax'd category: for category c,
predicted_avg_prob[c] = mean over every held-out row of that row's own
p_model(c | situation). This equals the expected fraction of rows that
would land in category c if each row's outcome were drawn once from the
model's own predicted distribution -- directly comparable to the real
observed count(c) / N, and a stricter check than comparing argmax-class
frequency would be (argmax-only can hide a systematically over/under-
confident model whose argmax choices still land on the right class most of
the time)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.contact_quality import DEFAULT_CHECKPOINT_PATH as DEFAULT_CONTACT_QUALITY_CHECKPOINT
from src.data.contact_quality import load_contact_quality_histories
from src.data.event_dataset import EventBatchCollator, EventDataset
from src.data.event_embedding_cache import DEFAULT_CACHE_DIR, EmbeddingCache
from src.data.park_factors import ParkFactorConfig, ParkFactorEmbedding, compute_league_rates, compute_park_factors
from src.data.sequence_dataset import OUTCOME_VOCAB
from src.data.statcast_common import PROCESSED_DATA_DIR, TRAIN_SEASON_RANGE, VAL_SEASONS, read_partitioned
from src.device import DEFAULT_DEVICE, resolve_device
from src.models.event_model import EventModel, EventModelConfig
from src.simulation.game_engine import log_checkpoint_training_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# How far predicted-vs-observed marginal probability can drift, as a ratio, before a category gets
# flagged rather than passed silently. Calibrated against the 2026-07 incident CLAUDE.md documents (an
# inverse-frequency-weighted run's predicted extra-base-hit rate came in ~3x real) -- a ratio at or beyond
# 2x is comfortably inside the range that incident would have tripped, without being so tight that
# ordinary sampling noise on rarer categories (e.g. triples) flags a genuinely fine checkpoint.
WARN_RATIO_THRESHOLD = 2.0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate marginal-probability calibration check: compares a trained EventModel "
        "checkpoint's average predicted probability per OUTCOME_VOCAB category against the real observed "
        "marginal frequency, over the checkpoint's own held-out validation split."
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to the event_model_*_best.pt checkpoint to validate.")
    parser.add_argument("--pitches-dir", type=Path, default=PROCESSED_DATA_DIR / "pitches")
    parser.add_argument("--embedding-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--embedding-cache-max-entries", type=int, default=None,
        help="Same LRU-eviction cap as train_event_model.py's flag of the same name -- see that script's "
        "help text. Leave unset to match whatever the checkpoint being validated was itself trained with, "
        "if known to have used unbounded caching.",
    )
    parser.add_argument("--contact-quality-checkpoint", type=Path, default=DEFAULT_CONTACT_QUALITY_CHECKPOINT)
    parser.add_argument(
        "--train-season-start", type=int, default=TRAIN_SEASON_RANGE[0],
        help="Must match the --train-season-start the checkpoint being validated was itself trained with -- "
        "used only to rebuild the park-factor embedding's (park_id, season) vocab identically (see "
        "game_engine.py's build_game_engine_context for the same convention/warning); a mismatch here "
        "produces a shape/index mismatch against the trained embedding table, not a silently-wrong result.",
    )
    parser.add_argument("--train-season-end", type=int, default=TRAIN_SEASON_RANGE[1])
    parser.add_argument(
        "--val-seasons", type=int, nargs="+", default=list(VAL_SEASONS),
        help="The held-out season(s) this check actually evaluates against -- must match the checkpoint's "
        "own --val-seasons for this to be a genuine held-out check.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    return parser.parse_args(argv)


def compute_marginal_calibration(
    model: EventModel, val_loader: DataLoader, device: torch.device,
) -> tuple[dict[str, float], dict[str, float], int]:
    """One eval-mode, no-grad pass over val_loader. Returns (predicted_avg_prob, observed_frequency, n),
    both dicts keyed by OUTCOME_VOCAB category name, n = total rows seen."""
    model.eval()
    prob_totals = torch.zeros(len(OUTCOME_VOCAB), dtype=torch.float64)
    observed_counts = torch.zeros(len(OUTCOME_VOCAB), dtype=torch.float64)
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            probs = F.softmax(logits, dim=-1)
            prob_totals += probs.sum(dim=0).double().cpu()
            observed_counts += torch.bincount(batch["target"], minlength=len(OUTCOME_VOCAB)).double().cpu()
            n += batch["target"].shape[0]

    predicted_avg_prob = {cat: (prob_totals[i] / n).item() for i, cat in enumerate(OUTCOME_VOCAB)}
    observed_frequency = {cat: (observed_counts[i] / n).item() for i, cat in enumerate(OUTCOME_VOCAB)}
    return predicted_avg_prob, observed_frequency, n


def main(argv=None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)

    logger.info("Loading event model checkpoint from %s", args.checkpoint_path)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    log_checkpoint_training_metadata(args.checkpoint_path, ckpt)

    model_config = EventModelConfig(**ckpt["model_config"])
    park_factor_config = ParkFactorConfig(**ckpt["park_factor_config"])
    situational_stats = ckpt["situational_stats"]
    contact_quality_stats = ckpt["contact_quality_stats"]

    train_season_range = (args.train_season_start, args.train_season_end)
    val_seasons = tuple(args.val_seasons)
    logger.info("Season split -- train: %d-%d, val: %s", *train_season_range, val_seasons)

    logger.info("Loading pitches from %s", args.pitches_dir)
    full = read_partitioned(args.pitches_dir)
    pitches = full[full["season"].between(train_season_range[0], val_seasons[-1]) & full["is_valid"]].reset_index(drop=True)
    del full
    val_pitches = pitches[pitches["season"].isin(val_seasons)].reset_index(drop=True)
    logger.info("Val pitches (real held-out situations): %d", len(val_pitches))

    # Rebuilt over the *same* train+val season range train_event_model.py used -- must match exactly, or
    # the reconstructed park-factor embedding's (park_id, season) vocab won't line up with the row indices
    # the trained embedding table in model_state_dict actually corresponds to (see --train-season-start's
    # help text, and game_engine.py's build_game_engine_context for the same convention).
    park_factors = compute_park_factors(pitches, rolling_years=park_factor_config.rolling_years)
    park_factor_embedding = ParkFactorEmbedding(park_factor_config, park_factors)
    league_rates = compute_league_rates(pitches, rolling_years=park_factor_config.rolling_years)
    del pitches

    logger.info("Loading contact-quality histories from %s", args.contact_quality_checkpoint)
    contact_quality = load_contact_quality_histories(args.contact_quality_checkpoint)
    pitcher_contact_quality, batter_contact_quality = contact_quality["pitcher"], contact_quality["batter"]

    val_dataset = EventDataset(
        val_pitches, situational_stats, park_factor_embedding, league_rates,
        pitcher_contact_quality, batter_contact_quality, contact_quality_stats,
    )
    del val_pitches

    pitcher_cache = EmbeddingCache(args.embedding_cache_dir, "pitcher", max_entries=args.embedding_cache_max_entries)
    batter_cache = EmbeddingCache(args.embedding_cache_dir, "batter", max_entries=args.embedding_cache_max_entries)
    collate_fn = EventBatchCollator(pitcher_cache, batter_cache)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers
    )

    model = EventModel(model_config, park_factor_embedding)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    predicted_avg_prob, observed_frequency, n = compute_marginal_calibration(model, val_loader, device)

    logger.info("Marginal calibration over %d real held-out pitches:", n)
    logger.info("%-14s %14s %14s %10s %8s", "category", "predicted_avg", "observed_freq", "abs_diff", "ratio")
    flagged = []
    for cat in OUTCOME_VOCAB:
        pred, obs = predicted_avg_prob[cat], observed_frequency[cat]
        abs_diff = pred - obs
        ratio = pred / obs if obs > 0 else (float("inf") if pred > 0 else 1.0)
        logger.info("%-14s %14.4f %14.4f %10.4f %8.2f", cat, pred, obs, abs_diff, ratio)
        if obs > 0 and (ratio >= WARN_RATIO_THRESHOLD or ratio <= 1 / WARN_RATIO_THRESHOLD):
            flagged.append((cat, pred, obs, ratio))

    if flagged:
        logger.warning(
            "%d/%d categories have predicted/observed ratio >= %.1fx or <= %.1fx -- this is the class of "
            "failure the marginal calibration check exists to catch (see CLAUDE.md's 2026-07 inverse-"
            "frequency-class-weighting incident: ~3x off in absolute terms while still passing relative "
            "simulation probes). Do NOT proceed to simulation-based validation until this is understood:",
            len(flagged), len(OUTCOME_VOCAB), WARN_RATIO_THRESHOLD, 1 / WARN_RATIO_THRESHOLD,
        )
        for cat, pred, obs, ratio in flagged:
            logger.warning("  %s: predicted=%.4f observed=%.4f ratio=%.2fx", cat, pred, obs, ratio)
    else:
        logger.info(
            "All %d categories within %.1fx of observed marginal frequency -- calibration looks reasonable, "
            "safe to proceed to simulation-based validation.", len(OUTCOME_VOCAB), WARN_RATIO_THRESHOLD,
        )


if __name__ == "__main__":
    main()
