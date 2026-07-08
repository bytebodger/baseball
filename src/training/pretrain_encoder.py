"""Pretrain PlayerEncoder on a "predict the next pitch's outcome" task.

For every pitcher's chronologically sorted pitch history, each pitch is one
training example: the pitches strictly before it (same pitcher, truncated to
the encoder's max_seq_len) are the input sequence, and its own outcome label
is the target. A pitcher's first clean pitch has no history at all, which
exercises PlayerEncoder's no-history embedding during pretraining rather than
leaving it untrained.

Data is split by season: train on 2015-2022, validate on 2023, and 2024-2025
are held out entirely (not loaded here) for later testing.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from src.data.sequence_dataset import (
    CONTINUOUS_FEATURES,
    MATCHUP_INDEX,
    OUTCOME_INDEX,
    OUTCOME_VOCAB,
    PITCH_TYPE_INDEX,
    PlayerPitchSequenceDataset,
    category_indices,
)
from src.data.statcast_common import (
    PROCESSED_DATA_DIR,
    TEST_SEASON_RANGE,
    TRAIN_SEASON_RANGE,
    VAL_SEASONS,
    read_partitioned,
)
from src.device import DEFAULT_DEVICE, resolve_device
from src.models.player_encoder import DEFAULT_CONFIG_PATH, PlayerEncoder, PlayerEncoderConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PITCHES_DIR = PROCESSED_DATA_DIR / "pitches"
EARLY_STOPPING_PATIENCE = 4


class NextPitchDataset(Dataset):
    """One sample per pitch: predict that pitch's outcome from the same
    pitcher's pitches strictly before it. Requires `pitches` to already be
    filtered to `is_valid` rows -- see load_season_split."""

    def __init__(
        self,
        pitches: pd.DataFrame,
        max_seq_len: int,
        continuous_stats: dict[str, tuple[float, float]],
    ) -> None:
        pitches = pitches.sort_values(
            ["pitcher_id", "game_date", "at_bat_number", "pitch_number"]
        ).reset_index(drop=True)

        self.max_seq_len = max_seq_len
        self.continuous_stats = continuous_stats

        # Row index where each row's pitcher's history begins: position in the
        # sorted frame minus that pitcher's running pitch count so far. Lets
        # __getitem__ slice a player's preceding history in O(window) time
        # instead of filtering the whole frame per sample.
        cumcount = pitches.groupby("pitcher_id").cumcount().to_numpy()
        self.group_start = np.arange(len(pitches)) - cumcount

        self.continuous = np.stack(
            [
                (pitches[col].to_numpy(dtype="float64", na_value=np.nan) - mean) / std
                for col, (mean, std) in continuous_stats.items()
            ],
            axis=1,
        )
        self.continuous = np.nan_to_num(self.continuous, nan=0.0).astype("float32")

        self.pitch_type_idx = category_indices(pitches["pitch_type"], PITCH_TYPE_INDEX).numpy()
        self.outcome_idx = category_indices(pitches["outcome"], OUTCOME_INDEX).numpy()
        matchup = pitches["stand"].astype(object) + "_" + pitches["p_throws"].astype(object)
        self.matchup_idx = category_indices(matchup, MATCHUP_INDEX).numpy()

    def __len__(self) -> int:
        return len(self.outcome_idx)

    def __getitem__(self, idx: int) -> dict:
        start = max(int(self.group_start[idx]), idx - self.max_seq_len)
        length = idx - start
        target = int(self.outcome_idx[idx])

        if length == 0:
            return {
                "has_history": False,
                "length": 0,
                "continuous": torch.zeros((0, len(CONTINUOUS_FEATURES)), dtype=torch.float32),
                "pitch_type": torch.zeros((0,), dtype=torch.long),
                "outcome": torch.zeros((0,), dtype=torch.long),
                "matchup": torch.zeros((0,), dtype=torch.long),
                "position": torch.zeros((0,), dtype=torch.long),
                "target": target,
            }

        return {
            "has_history": True,
            "length": length,
            "continuous": torch.from_numpy(self.continuous[start:idx]),
            "pitch_type": torch.from_numpy(self.pitch_type_idx[start:idx]),
            "outcome": torch.from_numpy(self.outcome_idx[start:idx]),
            "matchup": torch.from_numpy(self.matchup_idx[start:idx]),
            "position": torch.arange(length, dtype=torch.long),
            "target": target,
        }


def collate_next_pitch_batch(batch: list[dict]) -> tuple[dict, torch.Tensor]:
    batch_size = len(batch)
    max_len = max(max(sample["length"] for sample in batch), 1)

    continuous = torch.zeros(batch_size, max_len, len(CONTINUOUS_FEATURES))
    pitch_type = torch.zeros(batch_size, max_len, dtype=torch.long)
    outcome = torch.zeros(batch_size, max_len, dtype=torch.long)
    matchup = torch.zeros(batch_size, max_len, dtype=torch.long)
    position = torch.zeros(batch_size, max_len, dtype=torch.long)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    has_history = torch.zeros(batch_size, dtype=torch.bool)
    targets = torch.zeros(batch_size, dtype=torch.long)

    for i, sample in enumerate(batch):
        length = sample["length"]
        has_history[i] = sample["has_history"]
        targets[i] = sample["target"]
        if length == 0:
            continue
        continuous[i, :length] = sample["continuous"]
        pitch_type[i, :length] = sample["pitch_type"]
        outcome[i, :length] = sample["outcome"]
        matchup[i, :length] = sample["matchup"]
        position[i, :length] = sample["position"]
        padding_mask[i, :length] = False

    inputs = {
        "continuous": continuous,
        "pitch_type": pitch_type,
        "outcome": outcome,
        "matchup": matchup,
        "position": position,
        "padding_mask": padding_mask,
        "has_history": has_history,
    }
    return inputs, targets


class NextPitchPredictor(nn.Module):
    def __init__(self, config: PlayerEncoderConfig) -> None:
        super().__init__()
        self.encoder = PlayerEncoder(config)
        self.classifier = nn.Linear(config.hidden_size, len(OUTCOME_VOCAB))

    def forward(self, inputs: dict) -> torch.Tensor:
        embedding = self.encoder(
            inputs["continuous"],
            inputs["pitch_type"],
            inputs["outcome"],
            inputs["matchup"],
            inputs["position"],
            inputs["padding_mask"],
            inputs["has_history"],
        )
        return self.classifier(embedding)


def naive_baseline_metrics(val_df: pd.DataFrame) -> tuple[float, float, str]:
    """Accuracy and cross-entropy of a naive baseline that always predicts the
    single most frequent outcome class in `val_df` -- the floor a trained
    model needs to beat. The cross-entropy uses the same eps-clipping as
    sklearn's log_loss (rather than a literal 0/1 probability) so it stays
    finite despite the baseline assigning ~0 probability to every other class.
    """
    counts = val_df["outcome"].value_counts()
    majority_class = counts.idxmax()
    accuracy = counts.max() / len(val_df)

    eps = np.finfo(np.float64).eps
    num_classes = len(OUTCOME_VOCAB)
    p_correct = 1.0 - eps * (num_classes - 1)
    cross_entropy = -(accuracy * np.log(p_correct) + (1 - accuracy) * np.log(eps))
    return accuracy, cross_entropy, majority_class


def load_season_split(pitches_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = read_partitioned(pitches_dir)
    train_df = full[
        full["season"].between(*TRAIN_SEASON_RANGE) & full["is_valid"]
    ].reset_index(drop=True)
    val_df = full[full["season"].isin(VAL_SEASONS) & full["is_valid"]].reset_index(drop=True)
    return train_df, val_df


def run_epoch(
    model: NextPitchPredictor,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
) -> tuple[float, float]:
    train = optimizer is not None
    model.train(mode=train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.set_grad_enabled(train):
        for inputs, targets in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = targets.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, targets)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == targets).sum().item()
            total_count += batch_size

    return total_loss / total_count, total_correct / total_count


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain PlayerEncoder on next-pitch-outcome prediction.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="PlayerEncoder YAML config.")
    parser.add_argument("--pitches-dir", type=Path, default=DEFAULT_PITCHES_DIR)
    parser.add_argument("--epochs", type=int, default=25, help="Upper bound on epochs; early stopping may end it sooner.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Stop if validation loss doesn't improve for this many consecutive epochs.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Defaults to cuda -- this project trains on GPU. Pass --device cpu to explicitly opt into a "
        "(much slower) CPU run instead of silently falling back to one.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Cap train/val rows to the most recent N (smoke-testing only).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    config = PlayerEncoderConfig.from_yaml(args.config)

    logger.info(
        "Season split -- train: %d-%d, val: %s, held out for later testing: %d-%d",
        *TRAIN_SEASON_RANGE,
        VAL_SEASONS,
        *TEST_SEASON_RANGE,
    )
    logger.info("Loading processed pitches from %s", args.pitches_dir)
    train_df, val_df = load_season_split(args.pitches_dir)

    if args.limit_rows:
        train_df = train_df.tail(args.limit_rows).reset_index(drop=True)
        val_df = val_df.tail(max(args.limit_rows // 5, 1)).reset_index(drop=True)

    logger.info("Train pitches: %d, Val pitches: %d", len(train_df), len(val_df))

    baseline_accuracy, baseline_loss, majority_class = naive_baseline_metrics(val_df)
    logger.info(
        "Naive baseline (always predict %r) -- val_loss=%.4f val_accuracy=%.4f",
        majority_class,
        baseline_loss,
        baseline_accuracy,
    )

    continuous_stats = PlayerPitchSequenceDataset._compute_continuous_stats(train_df)

    train_dataset = NextPitchDataset(train_df, config.max_seq_len, continuous_stats)
    val_dataset = NextPitchDataset(val_df, config.max_seq_len, continuous_stats)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_next_pitch_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_next_pitch_batch,
        num_workers=args.num_workers,
    )

    model = NextPitchPredictor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / "pretrain_encoder.csv"
    checkpoint_path = args.checkpoint_dir / "player_encoder_best.pt"

    write_header = not log_path.exists()
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_without_improvement = 0

    with open(log_path, "a", newline="") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(
                model, train_loader, device, criterion, optimizer=optimizer, scaler=scaler, use_amp=use_amp
            )
            val_loss, val_acc = run_epoch(model, val_loader, device, criterion, use_amp=use_amp)

            logger.info(
                "Epoch %d/%d - train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                epoch,
                args.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
            log_file.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save(
                    {
                        "encoder_state_dict": model.encoder.state_dict(),
                        "classifier_state_dict": model.classifier.state_dict(),
                        "config": asdict(config),
                        "continuous_stats": continuous_stats,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                    },
                    checkpoint_path,
                )
                logger.info("Saved new best checkpoint to %s (val_loss=%.4f)", checkpoint_path, val_loss)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    logger.info(
                        "Early stopping: val_loss hasn't improved for %d consecutive epochs.",
                        epochs_without_improvement,
                    )
                    break

    logger.info(
        "Done. Model val_loss=%.4f val_accuracy=%.4f | Naive baseline val_loss=%.4f val_accuracy=%.4f",
        best_val_loss,
        best_val_acc,
        baseline_loss,
        baseline_accuracy,
    )


if __name__ == "__main__":
    main()
