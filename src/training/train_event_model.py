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
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.event_dataset import CONTEXT_DIM, EventBatchCollator, EventDataset, compute_situational_stats
from src.data.event_embedding_cache import DEFAULT_CACHE_DIR, EmbeddingCache
from src.data.park_factors import ParkFactorConfig, ParkFactorEmbedding, compute_league_rates, compute_park_factors
from src.data.statcast_common import PROCESSED_DATA_DIR, TRAIN_SEASON_RANGE, VAL_SEASONS, read_partitioned
from src.device import DEFAULT_DEVICE, resolve_device
from src.models.event_model import EventModel, EventModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TRAINING_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "train_event_model.yaml"
EARLY_STOPPING_PATIENCE = 4


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

    def to_event_model_config(self, player_embed_dim: int, include_context: bool) -> EventModelConfig:
        return EventModelConfig(
            player_embed_dim=player_embed_dim,
            matchup_embed_dim=self.matchup_embed_dim,
            park_factor_embed_dim=self.park_factor_embed_dim,
            situational_dim=CONTEXT_DIM,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            include_context=include_context,
        )


def _infer_player_embed_dim(cache: EmbeddingCache, pitches) -> int:
    """Reads the embedding width straight off one real cached entry, rather
    than from the LongHistoryEncoder checkpoint -- EventModel never touches
    the encoder itself, only its precomputed output, so this script doesn't
    need to know anything about the encoder's own architecture."""
    row = pitches.iloc[0]
    return int(cache.get(row["pitcher_id"], row["game_date"]).shape[-1])


def compute_loss_and_metrics(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    loss = F.cross_entropy(logits.float(), target)
    with torch.no_grad():
        accuracy = (logits.argmax(dim=-1) == target).float().mean().item()
    return loss, {"accuracy": accuracy}


def run_epoch(
    model: EventModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(mode=train)

    totals = {"loss": 0.0, "accuracy": 0.0}
    total_count = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(batch)

            loss, metrics = compute_loss_and_metrics(logits, batch["target"])

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = batch["target"].size(0)
            totals["loss"] += loss.item() * batch_size
            totals["accuracy"] += metrics["accuracy"] * batch_size
            total_count += batch_size

    return {k: v / total_count for k, v in totals.items()}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EventModel on the precomputed long-history embedding cache.")
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG_PATH)
    parser.add_argument("--pitches-dir", type=Path, default=PROCESSED_DATA_DIR / "pitches")
    parser.add_argument("--embedding-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
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
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
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
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    training_config = EventTrainingConfig.from_yaml(args.training_config)
    include_context = not args.no_context

    logger.info("Season split -- train: %d-%d, val: %s", *TRAIN_SEASON_RANGE, VAL_SEASONS)
    logger.info("Loading pitches from %s", args.pitches_dir)
    full = read_partitioned(args.pitches_dir)
    pitches = full[full["season"].between(TRAIN_SEASON_RANGE[0], VAL_SEASONS[-1]) & full["is_valid"]].reset_index(drop=True)

    train_pitches = pitches[pitches["season"].between(*TRAIN_SEASON_RANGE)].reset_index(drop=True)
    val_pitches = pitches[pitches["season"].isin(VAL_SEASONS)].reset_index(drop=True)

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

    train_dataset = EventDataset(train_pitches, situational_stats, park_factor_embedding, league_rates)
    val_dataset = EventDataset(val_pitches, situational_stats, park_factor_embedding, league_rates)
    collate_fn = EventBatchCollator(pitcher_cache, batter_cache)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers
    )

    model_config = training_config.to_event_model_config(player_embed_dim, include_context)
    model = EventModel(model_config, park_factor_embedding if include_context else None).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.lr)
    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = "no_context" if args.no_context else "full"
    log_path = args.log_dir / f"train_event_model_{suffix}.csv"
    checkpoint_path = args.checkpoint_dir / f"event_model_{suffix}_best.pt"

    write_header = not log_path.exists()
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    with open(log_path, "a", newline="") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer, scaler=scaler, use_amp=use_amp)
            val_metrics = run_epoch(model, val_loader, device, use_amp=use_amp)

            logger.info(
                "Epoch %d/%d (include_context=%s) - train_loss=%.4f train_accuracy=%.4f | val_loss=%.4f val_accuracy=%.4f",
                epoch, args.epochs, include_context,
                train_metrics["loss"], train_metrics["accuracy"], val_metrics["loss"], val_metrics["accuracy"],
            )
            writer.writerow(
                [epoch, train_metrics["loss"], train_metrics["accuracy"], val_metrics["loss"], val_metrics["accuracy"]]
            )
            log_file.flush()

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": asdict(model_config),
                        "park_factor_config": asdict(park_factor_config),
                        "situational_stats": situational_stats,
                        "epoch": epoch,
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
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
