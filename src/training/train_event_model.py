"""Trains EventModel (Phase 4) on the precomputed long-history embedding
cache (src/data/event_embedding_cache.py) plus situational/park/league
context (src/data/event_dataset.py).

Single-stage only: unlike train_game_predictor.py, there's no encoder to
freeze/unfreeze here -- the pitcher/batter embeddings EventModel consumes
were already computed once by the frozen LongHistoryEncoder and cached to
disk, so this script only ever trains the lightweight combiner (plus
ParkFactorEmbedding and a small matchup embedding, both real submodules of
EventModel when include_context=True) on top of them.

--no-context runs the ablation EventModelConfig.include_context is built
for: strip situational/park/league context, keeping only the two player
embeddings. Run this script twice (with and without --no-context) and
compare each run's best val_loss/val_accuracy to see how much of the
model's predictive power actually comes from situational context rather
than just knowing who's on the mound and at the plate.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.contact_quality import DEFAULT_CHECKPOINT_PATH as DEFAULT_CONTACT_QUALITY_CHECKPOINT
from src.data.contact_quality import load_contact_quality_histories
from src.data.event_dataset import (
    CONTEXT_DIM,
    EventBatchCollator,
    EventDataset,
    compute_contact_quality_stats,
    compute_situational_stats,
)
from src.data.event_embedding_cache import DEFAULT_CACHE_DIR, EmbeddingCache
from src.data.park_factors import ParkFactorConfig, ParkFactorEmbedding, compute_league_rates, compute_park_factors
from src.data.sequence_dataset import OUTCOME_VOCAB
from src.data.statcast_common import PROCESSED_DATA_DIR, TRAIN_SEASON_RANGE, VAL_SEASONS, read_partitioned
from src.device import DEFAULT_DEVICE, resolve_device
from src.models.event_model import EventModel, EventModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TRAINING_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "train_event_model.yaml"
EARLY_STOPPING_PATIENCE = 4

# The shared, canonical checkpoint location game_engine.py's DEFAULT_EVENT_MODEL_CHECKPOINT loads from by
# default. A 2026-07 incident: an aux-loss-weight run had an incompatible architecture vs. the checkpoint
# already sitting here, the resume-staleness check below correctly refused to resume from it, but nothing
# then stopped that run from still training from fresh init and overwriting this same shared path with its
# own (unintended) result -- silently replacing the "keeper" checkpoint. See --save-as-default below, which
# closes that off structurally rather than relying on someone noticing after the fact via training_metadata.
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
EXPERIMENTAL_CHECKPOINT_SUBDIR = "experimental"


@dataclass
class EventTrainingConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    matchup_embed_dim: int = 8
    park_factor_embed_dim: int = 8
    park_factor_rolling_years: int = 3
    lr: float = 1e-3

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_TRAINING_CONFIG_PATH) -> "EventTrainingConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_event_model_config(
        self, player_embed_dim: int, include_context: bool, interaction_type: str = "none", interaction_dim: int = 32
    ) -> EventModelConfig:
        return EventModelConfig(
            player_embed_dim=player_embed_dim,
            matchup_embed_dim=self.matchup_embed_dim,
            park_factor_embed_dim=self.park_factor_embed_dim,
            situational_dim=CONTEXT_DIM,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            include_context=include_context,
            interaction_type=interaction_type,
            interaction_dim=interaction_dim,
        )


def _infer_player_embed_dim(cache: EmbeddingCache, pitches) -> int:
    """Reads the embedding width straight off one real cached entry, rather
    than from the LongHistoryEncoder checkpoint -- EventModel never touches
    the encoder itself, only its precomputed output, so this script doesn't
    need to know anything about the encoder's own architecture."""
    row = pitches.iloc[0]
    return int(cache.get(row["pitcher_id"], row["game_date"]).shape[-1])


def compute_class_weights(target: torch.Tensor, num_classes: int, max_weight_ratio: float = 20.0) -> torch.Tensor:
    """Inverse-frequency weights for F.cross_entropy's `weight=` argument,
    normalized to mean 1 across all `num_classes` categories -- keeps the
    overall loss magnitude comparable to plain unweighted CE, rather than
    an unexplained scale shift that would also throw off cross-run best-
    val-loss comparisons (early stopping compares val_loss run over run).

    This project's real OUTCOME_VOCAB frequencies span roughly a 300x range
    (ball ~2.7M examples vs. triple ~8k) -- see the diagnostic that led here
    for why plain unweighted CE under-learns the rare, compound categories
    (double/triple/home_run) that determine contact-quality suppression.
    Raw inverse frequency is clipped so the largest weight is at most
    `max_weight_ratio` times the smallest: an *uncapped* inverse-frequency
    weight would give the rarest class a gradient contribution ~300x any
    single common-class example's, risking large, destabilizing gradient
    spikes early in training rather than the intended effect of just no
    longer being drowned out.
    """
    counts = torch.bincount(target, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)  # a class absent from this split gets a large but finite weight, not inf
    min_allowed_count = counts.max() / max_weight_ratio
    counts = torch.clamp(counts, min=min_allowed_count)
    weights = 1.0 / counts
    return weights * (num_classes / weights.sum())


def compute_loss_and_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    aux_predictions: torch.Tensor | None = None,
    aux_targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
    """Returns (main_loss, aux_loss_or_None, metrics). `metrics` always has
    "accuracy" and "aux_loss" (0.0 when the auxiliary head isn't in use) so
    run_epoch's running totals don't need to branch on whether it was
    computed this batch. aux_loss (MSE, see EventModel.contact_quality_aux_head's
    module docstring) is reported separately from main_loss rather than
    folded into one number -- the two are on different objectives
    (13-way classification vs. 2-way regression) and keeping them apart
    lets a caller decide how (or whether) to combine them into a backward
    target, without losing the individual values for logging."""
    main_loss = F.cross_entropy(logits.float(), target, weight=class_weights)
    aux_loss = None
    aux_loss_value = 0.0
    if aux_predictions is not None and aux_targets is not None:
        aux_loss = F.mse_loss(aux_predictions.float(), aux_targets.float())
        aux_loss_value = aux_loss.item()
    with torch.no_grad():
        accuracy = (logits.argmax(dim=-1) == target).float().mean().item()
    return main_loss, aux_loss, {"accuracy": accuracy, "aux_loss": aux_loss_value}


def run_epoch(
    model: EventModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
    class_weights: torch.Tensor | None = None,
    include_context: bool = True,
    aux_loss_weight: float = 0.0,
) -> dict[str, float]:
    """`include_context` gates whether contact_quality_aux_head even exists
    (EventModel.forward(..., return_aux=True) raises if include_context is
    False -- see that module's docstring) -- when False, no auxiliary loss
    is computed at all, regardless of `aux_loss_weight`.

    The auxiliary MSE loss is always *computed and reported* (as
    "aux_loss" in the returned dict) whenever include_context is True, so
    its trend is visible in the training log even if `aux_loss_weight=0`
    (an "off switch" for how much it steers gradients, not for whether it's
    tracked). The quantity actually backpropagated during training is
    main_loss + aux_loss_weight * aux_loss; the *reported* "loss" stays
    main_loss only, matching every prior run's val_loss/early-stopping
    semantics (same "don't let an auxiliary signal redefine what
    'best checkpoint' means" reasoning as class_weights not touching val
    loss either)."""
    train = optimizer is not None
    model.train(mode=train)

    totals = {"loss": 0.0, "accuracy": 0.0, "aux_loss": 0.0}
    total_count = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if include_context:
                    logits, aux_predictions = model(batch, return_aux=True)
                else:
                    logits, aux_predictions = model(batch), None

            main_loss, aux_loss, metrics = compute_loss_and_metrics(
                logits, batch["target"], class_weights, aux_predictions, batch.get("contact_quality_aux_target")
            )

            if train:
                total_loss = main_loss if aux_loss is None else main_loss + aux_loss_weight * aux_loss
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = batch["target"].size(0)
            totals["loss"] += main_loss.item() * batch_size
            totals["accuracy"] += metrics["accuracy"] * batch_size
            totals["aux_loss"] += metrics["aux_loss"] * batch_size
            total_count += batch_size

    return {k: v / total_count for k, v in totals.items()}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EventModel on the precomputed long-history embedding cache.")
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG_PATH)
    parser.add_argument("--pitches-dir", type=Path, default=PROCESSED_DATA_DIR / "pitches")
    parser.add_argument("--embedding-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--contact-quality-checkpoint", type=Path, default=DEFAULT_CONTACT_QUALITY_CHECKPOINT)
    parser.add_argument(
        "--train-season-start", type=int, default=TRAIN_SEASON_RANGE[0],
        help="Overrides the project-wide default train split start (statcast_common.TRAIN_SEASON_RANGE) -- "
        "e.g. for walk-forward retraining at a later season boundary.",
    )
    parser.add_argument(
        "--train-season-end", type=int, default=TRAIN_SEASON_RANGE[1],
        help="Overrides the project-wide default train split end (statcast_common.TRAIN_SEASON_RANGE).",
    )
    parser.add_argument(
        "--val-seasons", type=int, nargs="+", default=list(VAL_SEASONS),
        help="Overrides the project-wide default validation season(s) (statcast_common.VAL_SEASONS).",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Upper bound on epochs; early stopping may end it sooner.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Stop if validation loss doesn't improve for this many consecutive epochs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes. Leave at 0 on Windows if it errors on startup -- some Windows "
        "Python installs (notably the Microsoft Store build) can't be re-spawned as a subprocess.",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--save-as-default",
        action="store_true",
        help=f"Required to let this run write into the shared default checkpoint directory "
        f"({DEFAULT_CHECKPOINT_DIR}) -- e.g. the canonical/keeper checkpoint game_engine.py loads by default. "
        f"Without this flag, a run left at the default --checkpoint-dir is automatically redirected to "
        f"{DEFAULT_CHECKPOINT_DIR / EXPERIMENTAL_CHECKPOINT_SUBDIR} instead, so an exploratory/experimental "
        f"run (different seed, aux-loss-weight, interaction-type, etc.) can never silently overwrite the "
        f"canonical checkpoint just by omitting --checkpoint-dir. Pass an explicit --checkpoint-dir to choose "
        f"your own location instead of either of these.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seeds torch's global RNG (weight init, dropout, and DataLoader shuffling all derive from it) so a "
        "run can be reproduced or deliberately varied. Data loading/splitting/feature computation are otherwise "
        "deterministic given the same inputs, so this is the only source of run-to-run variance for a fixed "
        "config.",
    )
    parser.add_argument(
        "--interaction-type",
        choices=["none", "bilinear", "film", "elementwise"],
        default="none",
        help="Explicit pitcher-batter interaction mechanism (see EventModelConfig/EventModel docstrings). "
        "'none' reproduces the original concatenation-only architecture exactly.",
    )
    parser.add_argument(
        "--interaction-dim",
        type=int,
        default=32,
        help="Width of the low-rank projected interaction vector -- only used by --interaction-type=bilinear.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Defaults to cuda -- this project trains on GPU. Pass --device cpu to explicitly opt into a "
        "(much slower) CPU run instead of silently falling back to one.",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Ablation: strip situational/park/league-rate context, keeping only the two player embeddings "
        "(EventModelConfig.include_context=False). Run once with this flag and once without to compare.",
    )
    parser.add_argument(
        "--limit-rows", type=int, default=None, help="Cap train/val pitches to the most recent N (smoke-testing only)."
    )
    parser.add_argument(
        "--class-weighted-loss",
        action="store_true",
        help="Opt-in: train with compute_class_weights' inverse-frequency-weighted cross-entropy instead of plain "
        "unweighted CE. Off by default -- an earlier attempt at this (uniform ~1.8x weight, capped at "
        "max_weight_ratio=20, on every rare OUTCOME_VOCAB category at once) pushed the model's absolute predicted "
        "extra-base-hit rate to ~3x real rates and made full-game simulations average ~59 runs/game instead of "
        "~9. Left available for future, more careful tuning (e.g. a lower cap, or reweighting only the specific "
        "categories that need it) rather than removed outright.",
    )
    parser.add_argument(
        "--aux-loss-weight",
        type=float,
        default=0.1,
        help="Weight on EventModel.contact_quality_aux_head's MSE loss (real, leak-safe pitcher BABIP-allowed/"
        "hard-hit-rate-allowed, see contact_quality.py), added to the main cross-entropy loss during training "
        "only (val_loss/early-stopping stay main-loss-only, same as --class-weighted-loss's own val-loss "
        "convention). 'Modest' by design -- this shares the trunk's own parameters with the main classification "
        "head, so it's meant to nudge the trunk's hidden representation toward encoding contact quality, not to "
        "dominate what the model optimizes for. Pass 0 to disable (the head still exists and its MSE is still "
        "logged, just never backpropagated). Only used when include_context is True.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    training_config = EventTrainingConfig.from_yaml(args.training_config)
    include_context = not args.no_context

    train_season_range = (args.train_season_start, args.train_season_end)
    val_seasons = tuple(args.val_seasons)
    logger.info("Season split -- train: %d-%d, val: %s", *train_season_range, val_seasons)
    logger.info("Loading pitches from %s", args.pitches_dir)
    full = read_partitioned(args.pitches_dir)
    pitches = full[full["season"].between(train_season_range[0], val_seasons[-1]) & full["is_valid"]].reset_index(drop=True)

    train_pitches = pitches[pitches["season"].between(*train_season_range)].reset_index(drop=True)
    val_pitches = pitches[pitches["season"].isin(val_seasons)].reset_index(drop=True)

    if args.limit_rows:
        train_pitches = train_pitches.tail(args.limit_rows).reset_index(drop=True)
        val_pitches = val_pitches.tail(max(args.limit_rows // 5, 1)).reset_index(drop=True)

    logger.info("Train pitches: %d, Val pitches: %d", len(train_pitches), len(val_pitches))

    situational_stats = compute_situational_stats(train_pitches)

    # Park factors/league rates computed over train+val together: each
    # season's rolling window only ever reaches strictly-prior seasons (see
    # park_factors.py), so this can't leak a train or val season's own
    # results into its own factor -- it just means val seasons get a real
    # (rather than fallback) factor too.
    park_factor_config = ParkFactorConfig(
        rolling_years=training_config.park_factor_rolling_years, embedding_dim=training_config.park_factor_embed_dim
    )
    park_factors = compute_park_factors(pitches, rolling_years=park_factor_config.rolling_years)
    park_factor_embedding = ParkFactorEmbedding(park_factor_config, park_factors)
    league_rates = compute_league_rates(pitches, rolling_years=park_factor_config.rolling_years)

    pitcher_cache = EmbeddingCache(args.embedding_cache_dir, "pitcher")
    batter_cache = EmbeddingCache(args.embedding_cache_dir, "batter")
    player_embed_dim = _infer_player_embed_dim(pitcher_cache, train_pitches)

    logger.info("Loading contact-quality histories from %s", args.contact_quality_checkpoint)
    contact_quality = load_contact_quality_histories(args.contact_quality_checkpoint)
    pitcher_contact_quality, batter_contact_quality = contact_quality["pitcher"], contact_quality["batter"]
    # Train-split-only stats (same leak-safe convention as situational_stats).
    contact_quality_stats = compute_contact_quality_stats(train_pitches, pitcher_contact_quality, batter_contact_quality)

    train_dataset = EventDataset(
        train_pitches, situational_stats, park_factor_embedding, league_rates,
        pitcher_contact_quality, batter_contact_quality, contact_quality_stats,
    )
    val_dataset = EventDataset(
        val_pitches, situational_stats, park_factor_embedding, league_rates,
        pitcher_contact_quality, batter_contact_quality, contact_quality_stats,
    )
    collate_fn = EventBatchCollator(pitcher_cache, batter_cache)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers
    )

    model_config = training_config.to_event_model_config(player_embed_dim, include_context, args.interaction_type, args.interaction_dim)
    model = EventModel(model_config, park_factor_embedding if include_context else None).to(device)

    # Train-split-only, inverse-frequency class weights (see
    # compute_class_weights) -- opt-in via --class-weighted-loss, applied to
    # the training loss only. Val loss stays plain/unweighted regardless,
    # since it drives early stopping and checkpoint selection, and an
    # artificially reweighted metric there would no longer represent real
    # predictive performance on realistic data.
    class_weights = None
    if args.class_weighted_loss:
        class_weights = compute_class_weights(train_dataset.target, len(OUTCOME_VOCAB)).to(device)
        logger.info("Class weights (OUTCOME_VOCAB order): %s", dict(zip(OUTCOME_VOCAB, class_weights.tolist())))
    else:
        logger.info("Training with plain unweighted cross-entropy (pass --class-weighted-loss to opt into inverse-frequency weighting).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.lr)
    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)

    if args.checkpoint_dir == DEFAULT_CHECKPOINT_DIR and not args.save_as_default:
        redirected_dir = DEFAULT_CHECKPOINT_DIR / EXPERIMENTAL_CHECKPOINT_SUBDIR
        logger.warning(
            "--checkpoint-dir was left at the shared default (%s) without --save-as-default -- redirecting "
            "this run's checkpoints to %s instead, so it can't silently overwrite the canonical checkpoint. "
            "Pass --save-as-default to intentionally write the canonical checkpoint, or pass an explicit "
            "--checkpoint-dir to choose your own location.",
            DEFAULT_CHECKPOINT_DIR, redirected_dir,
        )
        args.checkpoint_dir = redirected_dir

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = "no_context" if args.no_context else "full"
    log_path = args.log_dir / f"train_event_model_{suffix}.csv"
    checkpoint_path = args.checkpoint_dir / f"event_model_{suffix}_best.pt"

    write_header = not log_path.exists()
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1

    # Resumable: a killed/interrupted run (this training loop can run close
    # to CLAUDE.md's ~30-minute background-task ceiling once contact-quality
    # feature construction is added) picks back up from its own last-saved
    # checkpoint on rerun with the exact same command, rather than starting
    # over from epoch 1 -- same "skip already-completed work" spirit as
    # src/resumable_job.py's run_until_complete, applied directly to this
    # training loop's own natural unit of progress (an epoch) instead of
    # through that module's generic remaining-work-count machinery.
    if checkpoint_path.exists():
        existing_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Only actually a resumable checkpoint if it matches this run's own
        # architecture -- checkpoint_path is a fixed filename shared across
        # every run of this script, so a checkpoint left over from a
        # *previous, differently-configured* run (a different context
        # feature set, hidden_dim, etc.) can genuinely be sitting there. A
        # shape mismatch means it's stale, not resumable, and this run
        # should just start fresh and overwrite it -- the same outcome a
        # pre-resume version of this script would have had.
        #
        # model_config equality alone isn't sufficient: EventModelConfig
        # doesn't have a field for every architectural choice (e.g. whether
        # contact_quality_aux_head exists is implied by include_context, not
        # its own config field) -- two runs can share an identical
        # model_config while producing state_dicts with different key sets.
        # That's exactly what happened going from the raw-scalar-only
        # architecture to this one: same config, new submodule, and a
        # config-only check would have missed it and crashed inside
        # load_state_dict instead of degrading gracefully to "start fresh."
        config_matches = existing_checkpoint.get("model_config") == asdict(model_config)
        keys_match = config_matches and set(existing_checkpoint.get("model_state_dict", {}).keys()) == set(model.state_dict().keys())
        if not keys_match:
            # Loud and printed directly (not just logged) *before training starts* -- this is exactly the
            # moment a run silently overwriting someone else's checkpoint needs to be caught, not after the
            # fact via training_metadata at load time.
            print(
                "\n" + "!" * 100 + "\n"
                f"FRESH INIT: the checkpoint already at {checkpoint_path} has a different architecture than "
                f"this run's (model_config and/or state_dict keys don't match) -- NOT resuming from it, "
                f"training from scratch instead. This run will still save its own result to that same path "
                f"when it finds a new best val_loss.\n" + "!" * 100 + "\n",
                flush=True,
            )
            logger.warning(
                "Found a checkpoint at %s, but its model_config and/or state_dict keys don't match this run's -- "
                "treating it as stale (from a differently-configured run) rather than resuming from it. Starting fresh.",
                checkpoint_path,
            )
        else:
            logger.info("Found existing checkpoint at %s -- resuming from it.", checkpoint_path)
            model.load_state_dict(existing_checkpoint["model_state_dict"])
            if "optimizer_state_dict" in existing_checkpoint:
                optimizer.load_state_dict(existing_checkpoint["optimizer_state_dict"])
            best_val_loss = existing_checkpoint["val_loss"]
            start_epoch = existing_checkpoint["epoch"] + 1
            logger.info("Resuming at epoch %d (best_val_loss so far=%.4f).", start_epoch, best_val_loss)
            if start_epoch > args.epochs:
                logger.info("Checkpoint's epoch %d already meets/exceeds --epochs=%d -- nothing left to do.", existing_checkpoint["epoch"], args.epochs)

    with open(log_path, "a", newline="") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(["epoch", "train_loss", "train_accuracy", "train_aux_loss", "val_loss", "val_accuracy", "val_aux_loss"])

        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, device, optimizer=optimizer, scaler=scaler, use_amp=use_amp,
                class_weights=class_weights, include_context=include_context, aux_loss_weight=args.aux_loss_weight,
            )
            val_metrics = run_epoch(model, val_loader, device, use_amp=use_amp, include_context=include_context)

            logger.info(
                "Epoch %d/%d (include_context=%s) - train_loss=%.4f train_accuracy=%.4f train_aux_loss=%.4f | "
                "val_loss=%.4f val_accuracy=%.4f val_aux_loss=%.4f",
                epoch, args.epochs, include_context,
                train_metrics["loss"], train_metrics["accuracy"], train_metrics["aux_loss"],
                val_metrics["loss"], val_metrics["accuracy"], val_metrics["aux_loss"],
            )
            writer.writerow(
                [epoch, train_metrics["loss"], train_metrics["accuracy"], train_metrics["aux_loss"],
                 val_metrics["loss"], val_metrics["accuracy"], val_metrics["aux_loss"]]
            )
            log_file.flush()

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_config": asdict(model_config),
                        "park_factor_config": asdict(park_factor_config),
                        "situational_stats": situational_stats,
                        "contact_quality_stats": contact_quality_stats,
                        "epoch": epoch,
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_aux_loss": val_metrics["aux_loss"],
                        "training_metadata": {
                            "aux_loss_weight": args.aux_loss_weight,
                            "seed": args.seed,
                            "class_weighted_loss": args.class_weighted_loss,
                            "include_context": include_context,
                            "contact_quality_feature_style": "raw-scalar, folded into context",
                            "interaction_type": args.interaction_type,
                            "interaction_dim": args.interaction_dim,
                            "saved_at": datetime.now(timezone.utc).isoformat(),
                        },
                    },
                    checkpoint_path,
                )
                logger.info("Saved new best checkpoint to %s (val_loss=%.4f)", checkpoint_path, best_val_loss)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    logger.info(
                        "Early stopping: val_loss hasn't improved for %d consecutive epochs.", epochs_without_improvement
                    )
                    break

    logger.info("Done (include_context=%s). Best val_loss=%.4f", include_context, best_val_loss)


if __name__ == "__main__":
    main()
