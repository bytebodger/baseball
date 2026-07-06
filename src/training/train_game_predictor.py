"""Trains GamePredictor end-to-end on GameOutcomeDataset (Phase 6), starting
from a PlayerEncoder checkpoint already pretrained by pretrain_encoder.py
(Phase 5).

Composes, per batch:
- the shared PlayerEncoder (from the Phase 5 checkpoint) encodes each game's
  home/away starting pitcher directly, and every home/away bullpen arm and
  lineup batter (flattened across the whole minibatch into one encoder call,
  then split back per game -- see _encode_and_pool);
- two PlayerSetPoolers (Phase 7), one for bullpens and one for lineups, pool
  each team's set of per-player embeddings into one fixed-size vector,
  handling a short bullpen or incomplete lineup via their learned
  empty-set embedding when a team has none available;
- GamePredictor (Phase 8) combines both starters' embeddings, both pooled
  bullpen/lineup embeddings, and a context-feature vector (month + starter
  rest days, cyclically/z-score encoded -- see _build_context_features) into
  a win-probability and a per-team runs prediction.

Whether/how the encoder trains is controlled entirely by the YAML training
config (see configs/train_game_predictor.yaml): freeze it outright, fine-tune
it jointly with the predictor from epoch 1 (at a separate, lower encoder_lr),
or fine-tune it in two stages (predictor-only warmup, then unfreeze).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.game_dataset import (
    BATTER_APPEARANCES_DIR,
    DEFAULT_BULLPEN_WINDOW_DAYS,
    DEFAULT_MAX_LINEUP_SIZE,
    GAMES_DIR,
    PITCHER_APPEARANCES_DIR,
    GameOutcomeDataset,
    load_game_split,
)
from src.data.sequence_dataset import CONTINUOUS_FEATURES, PlayerPitchSequenceDataset
from src.data.statcast_common import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TEST_SEASON_RANGE,
    TRAIN_SEASON_RANGE,
    VAL_SEASONS,
    read_partitioned,
)
from src.models.game_predictor import GamePredictor, GamePredictorConfig, NegativeBinomialHead
from src.models.player_encoder import DEFAULT_CONFIG_PATH as PLAYER_ENCODER_CONFIG_PATH
from src.models.player_encoder import PlayerEncoder, PlayerEncoderConfig
from src.models.set_pooling import DEFAULT_CONFIG_PATH as SET_POOLING_CONFIG_PATH
from src.models.set_pooling import PlayerSetPooler, PlayerSetPoolerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TRAINING_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "train_game_predictor.yaml"
DEFAULT_ENCODER_CHECKPOINT_PATH = Path("checkpoints") / "player_encoder_best.pt"
DEFAULT_SEQUENCE_CACHE_DIR = PROCESSED_DATA_DIR / "sequence_cache"
EARLY_STOPPING_PATIENCE = 4

# month sin/cos + {home,away} rest-days (z-scored, with a missing-indicator
# each for a pitcher's first career start, where rest_days is NaN).
CONTEXT_DIM = 6


@dataclass
class GamePredictorTrainingConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    runs_distribution: str = "negative_binomial"
    freeze_encoder: bool = False
    training_mode: str = "joint"  # "joint" or "two_stage"
    stage1_epochs: int = 5
    encoder_lr: float = 1e-5
    predictor_lr: float = 1e-3

    def __post_init__(self) -> None:
        if self.training_mode not in {"joint", "two_stage"}:
            raise ValueError(f"training_mode must be 'joint' or 'two_stage', got {self.training_mode!r}")

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_TRAINING_CONFIG_PATH) -> "GamePredictorTrainingConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_game_predictor_config(self) -> GamePredictorConfig:
        return GamePredictorConfig(
            context_dim=CONTEXT_DIM,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            runs_distribution=self.runs_distribution,
            freeze_player_encoder=False,  # freezing is managed per-epoch by this script instead
        )


class GamePredictionSystem(nn.Module):
    """Wires the shared PlayerEncoder (owned by `game_predictor`) together
    with a bullpen pooler and a lineup pooler so the whole pipeline -- from
    raw per-player pitch sequences to a game prediction -- trains as one
    model."""

    def __init__(self, game_predictor: GamePredictor, bullpen_pooler: PlayerSetPooler, lineup_pooler: PlayerSetPooler) -> None:
        super().__init__()
        self.game_predictor = game_predictor
        self.bullpen_pooler = bullpen_pooler
        self.lineup_pooler = lineup_pooler

    def forward(self, batch: dict) -> dict:
        player_encoder = self.game_predictor.player_encoder
        home_bullpen = _encode_and_pool(batch["home_bullpen"], batch["home_bullpen_set_sizes"], player_encoder, self.bullpen_pooler)
        away_bullpen = _encode_and_pool(batch["away_bullpen"], batch["away_bullpen_set_sizes"], player_encoder, self.bullpen_pooler)
        home_lineup = _encode_and_pool(batch["home_lineup"], batch["home_lineup_set_sizes"], player_encoder, self.lineup_pooler)
        away_lineup = _encode_and_pool(batch["away_lineup"], batch["away_lineup_set_sizes"], player_encoder, self.lineup_pooler)

        return self.game_predictor(
            batch["home_starter"], batch["away_starter"],
            home_bullpen, away_bullpen,
            home_lineup, away_lineup,
            batch["context"],
        )


def _encode_and_pool(
    flat_sequences: dict[str, torch.Tensor],
    set_sizes: torch.Tensor,
    player_encoder: PlayerEncoder,
    pooler: PlayerSetPooler,
) -> torch.Tensor:
    """flat_sequences holds every set member (bullpen arm or lineup batter)
    across the whole minibatch, flattened into one padded batch (see
    _pad_player_sequences); set_sizes says how many of them belong to each
    game so the per-player embeddings can be split back up and pooled."""
    if set_sizes.sum().item() == 0:
        # PlayerEncoder.forward can't run on a zero-row batch (its
        # has_history/no_history split assumes at least one row), and there's
        # nothing to encode anyway -- every game in this minibatch has an
        # empty set here (e.g. an all-empty bullpen).
        flat_embeddings = torch.zeros(0, pooler.config.embed_dim, device=set_sizes.device)
    else:
        flat_embeddings = player_encoder(**flat_sequences)

    per_game_embeddings = list(torch.split(flat_embeddings, set_sizes.tolist()))
    padded, padding_mask = PlayerSetPooler.pad_embeddings(per_game_embeddings)
    return pooler(padded, padding_mask)


def _pad_player_sequences(sequences: list[dict]) -> dict[str, torch.Tensor]:
    """Pads a flat list of per-player pitch-history sequences (each as
    returned by PlayerPitchSequenceDataset.build_sequence) into the batch
    tensors PlayerEncoder.forward expects."""
    batch_size = len(sequences)
    max_len = max(max((s["length"] for s in sequences), default=0), 1)

    continuous = torch.zeros(batch_size, max_len, len(CONTINUOUS_FEATURES))
    pitch_type = torch.zeros(batch_size, max_len, dtype=torch.long)
    outcome = torch.zeros(batch_size, max_len, dtype=torch.long)
    matchup = torch.zeros(batch_size, max_len, dtype=torch.long)
    position = torch.zeros(batch_size, max_len, dtype=torch.long)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    has_history = torch.zeros(batch_size, dtype=torch.bool)

    for i, sample in enumerate(sequences):
        length = sample["length"]
        has_history[i] = sample["has_history"]
        if length == 0:
            continue
        continuous[i, :length] = sample["continuous"]
        pitch_type[i, :length] = sample["pitch_type"]
        outcome[i, :length] = sample["outcome"]
        matchup[i, :length] = sample["matchup"]
        position[i, :length] = sample["position"]
        padding_mask[i, :length] = False

    return {
        "continuous": continuous,
        "pitch_type": pitch_type,
        "outcome": outcome,
        "matchup": matchup,
        "position": position,
        "padding_mask": padding_mask,
        "has_history": has_history,
    }


def _flatten_and_pad_sets(sets: list[list[dict]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    flat = [sequence for game_set in sets for sequence in game_set]
    set_sizes = torch.tensor([len(game_set) for game_set in sets], dtype=torch.long)
    return _pad_player_sequences(flat), set_sizes


def _build_context_features(
    month: torch.Tensor, home_rest: torch.Tensor, away_rest: torch.Tensor, rest_mean: float, rest_std: float
) -> torch.Tensor:
    angle = month / 12.0 * 2 * math.pi
    home_missing = torch.isnan(home_rest).float()
    away_missing = torch.isnan(away_rest).float()
    home_norm = torch.nan_to_num((home_rest - rest_mean) / rest_std, nan=0.0)
    away_norm = torch.nan_to_num((away_rest - rest_mean) / rest_std, nan=0.0)
    return torch.stack([torch.sin(angle), torch.cos(angle), home_norm, home_missing, away_norm, away_missing], dim=1)


class GameBatchCollator:
    """Turns a list of GameOutcomeDataset samples into one training batch:
    pads each side's starting-pitcher sequence, flattens+pads each side's
    bullpen/lineup members (see _flatten_and_pad_sets), and assembles the
    context-feature vector. rest_day_mean/std must come from the training
    split only (see _compute_rest_day_stats), same no-leakage spirit as
    PlayerPitchSequenceDataset's own continuous-feature normalization."""

    def __init__(self, rest_day_mean: float, rest_day_std: float) -> None:
        self.rest_day_mean = rest_day_mean
        self.rest_day_std = rest_day_std

    def __call__(self, batch: list[dict]) -> dict:
        home_bullpen, home_bullpen_sizes = _flatten_and_pad_sets([g["home_bullpen"] for g in batch])
        away_bullpen, away_bullpen_sizes = _flatten_and_pad_sets([g["away_bullpen"] for g in batch])
        home_lineup, home_lineup_sizes = _flatten_and_pad_sets([g["home_lineup"] for g in batch])
        away_lineup, away_lineup_sizes = _flatten_and_pad_sets([g["away_lineup"] for g in batch])

        month = torch.tensor([g["month"] for g in batch], dtype=torch.float32)
        home_rest = torch.tensor([g["home_starter_rest_days"] for g in batch], dtype=torch.float32)
        away_rest = torch.tensor([g["away_starter_rest_days"] for g in batch], dtype=torch.float32)

        return {
            "home_starter": _pad_player_sequences([g["home_starter"] for g in batch]),
            "away_starter": _pad_player_sequences([g["away_starter"] for g in batch]),
            "home_bullpen": home_bullpen,
            "home_bullpen_set_sizes": home_bullpen_sizes,
            "away_bullpen": away_bullpen,
            "away_bullpen_set_sizes": away_bullpen_sizes,
            "home_lineup": home_lineup,
            "home_lineup_set_sizes": home_lineup_sizes,
            "away_lineup": away_lineup,
            "away_lineup_set_sizes": away_lineup_sizes,
            "context": _build_context_features(month, home_rest, away_rest, self.rest_day_mean, self.rest_day_std),
            "home_score": torch.tensor([g["home_score"] for g in batch], dtype=torch.float32),
            "away_score": torch.tensor([g["away_score"] for g in batch], dtype=torch.float32),
            "home_win": torch.tensor([float(g["home_win"]) for g in batch], dtype=torch.float32),
        }


def _compute_rest_day_stats(train_games: pd.DataFrame) -> tuple[float, float]:
    values = pd.concat([train_games["home_starter_rest_days"], train_games["away_starter_rest_days"]]).dropna()
    if len(values) == 0:
        return 0.0, 1.0
    std = float(values.std())
    return float(values.mean()), std if std > 1e-6 else 1.0


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {k: v.to(device) for k, v in value.items()}
        else:
            moved[key] = value.to(device)
    return moved


def load_pretrained_encoder(checkpoint_path: Path) -> tuple[PlayerEncoder, dict[str, tuple[float, float]]]:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = PlayerEncoderConfig(**checkpoint["config"])
    encoder = PlayerEncoder(config)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    return encoder, checkpoint["continuous_stats"]


def build_random_encoder(config_path: Path, pitches_dir: Path) -> tuple[PlayerEncoder, dict[str, tuple[float, float]]]:
    """An ablation path for --no-pretrained-encoder: same architecture a
    pretrained checkpoint would have (read from the same YAML pretrain_encoder.py
    itself uses), but randomly initialized instead of loaded, so GamePredictor's
    own training is the *only* signal the encoder ever sees. continuous_stats
    still has to come from somewhere since sequence normalization doesn't
    depend on encoder weights -- computed the same way pretrain_encoder.py's
    checkpoint would have (train-season, is_valid pitches only), so it's
    numerically identical to what a pretrained checkpoint would carry and
    the two runs' sequence caches stay interchangeable.
    """
    config = PlayerEncoderConfig.from_yaml(config_path)
    encoder = PlayerEncoder(config)

    full_pitches = read_partitioned(pitches_dir)
    train_pitches = full_pitches[full_pitches["season"].between(*TRAIN_SEASON_RANGE) & full_pitches["is_valid"]]
    continuous_stats = PlayerPitchSequenceDataset._compute_continuous_stats(train_pitches)
    return encoder, continuous_stats


def _load_datasets(
    pitches_dir: Path,
    max_seq_len: int,
    bullpen_window_days: int,
    max_lineup_size: int,
    continuous_stats: dict[str, tuple[float, float]],
    raw_dir: Path,
    games_dir: Path,
    pitcher_appearances_dir: Path,
    batter_appearances_dir: Path,
    cache_dir: Path | None = None,
) -> tuple[GameOutcomeDataset, GameOutcomeDataset, pd.DataFrame]:
    """Mirrors game_dataset.load_train_val_game_datasets, but reuses the
    pretrained encoder's own continuous_stats instead of recomputing them --
    sequence features must be normalized exactly as the encoder was
    pretrained on, or its weights see out-of-distribution inputs."""
    full_pitches = read_partitioned(pitches_dir)
    pitches = full_pitches[
        full_pitches["season"].between(TRAIN_SEASON_RANGE[0], VAL_SEASONS[-1]) & full_pitches["is_valid"]
    ].reset_index(drop=True)

    train_games, val_games, pitcher_appearances, batter_appearances = load_game_split(
        raw_dir=raw_dir,
        games_dir=games_dir,
        pitcher_appearances_dir=pitcher_appearances_dir,
        batter_appearances_dir=batter_appearances_dir,
    )

    train_dataset = GameOutcomeDataset(
        pitches, train_games, pitcher_appearances, batter_appearances,
        max_seq_len, bullpen_window_days, max_lineup_size, continuous_stats, cache_dir,
    )
    val_dataset = GameOutcomeDataset(
        pitches, val_games, pitcher_appearances, batter_appearances,
        max_seq_len, bullpen_window_days, max_lineup_size, continuous_stats, cache_dir,
    )
    return train_dataset, val_dataset, train_games


def _encoder_trainable_this_epoch(epoch: int, config: GamePredictorTrainingConfig) -> bool:
    if config.freeze_encoder:
        return False
    if config.training_mode == "two_stage":
        return epoch > config.stage1_epochs
    return True


def _set_encoder_trainable(system: GamePredictionSystem, trainable: bool) -> None:
    encoder = system.game_predictor.player_encoder
    for param in encoder.parameters():
        param.requires_grad_(trainable)
    if not trainable:
        encoder.eval()


def compute_loss_and_metrics(output: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
    win_logit = output["win_logit"].float()
    win_loss = F.binary_cross_entropy_with_logits(win_logit, batch["home_win"])

    home_mean = output["home_runs"]["mean"].float()
    away_mean = output["away_runs"]["mean"].float()

    if "total_count" in output["home_runs"]:
        home_dist = NegativeBinomialHead.to_distribution(home_mean, output["home_runs"]["total_count"].float())
        away_dist = NegativeBinomialHead.to_distribution(away_mean, output["away_runs"]["total_count"].float())
        runs_loss = -(home_dist.log_prob(batch["home_score"]).mean() + away_dist.log_prob(batch["away_score"]).mean())
    else:
        runs_loss = F.mse_loss(home_mean, batch["home_score"]) + F.mse_loss(away_mean, batch["away_score"])

    loss = win_loss + runs_loss

    with torch.no_grad():
        win_prob = torch.sigmoid(win_logit)
        brier = ((win_prob - batch["home_win"]) ** 2).mean().item()
        home_mae = (home_mean - batch["home_score"]).abs().mean().item()
        away_mae = (away_mean - batch["away_score"]).abs().mean().item()

    return loss, {"brier": brier, "home_mae": home_mae, "away_mae": away_mae}


def run_epoch(
    system: GamePredictionSystem,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
    encoder_trainable: bool = True,
) -> dict[str, float]:
    train = optimizer is not None
    system.train(mode=train)
    if train:
        _set_encoder_trainable(system, encoder_trainable)

    totals = {"loss": 0.0, "brier": 0.0, "home_mae": 0.0, "away_mae": 0.0}
    total_count = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = _move_batch_to_device(batch, device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = system(batch)

            # Cast back to float32 before the loss: the negative-binomial
            # log-prob's lgamma terms and BCE's log terms are numerically
            # sensitive, so autocast's fp16 speedup is only worth taking on
            # the encoder/attention/trunk matmuls, not here.
            loss, metrics = compute_loss_and_metrics(output, batch)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = batch["home_win"].size(0)
            totals["loss"] += loss.item() * batch_size
            totals["brier"] += metrics["brier"] * batch_size
            totals["home_mae"] += metrics["home_mae"] * batch_size
            totals["away_mae"] += metrics["away_mae"] * batch_size
            total_count += batch_size

    return {k: v / total_count for k, v in totals.items()}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GamePredictor end-to-end from a Phase 5 pretrained PlayerEncoder checkpoint."
    )
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG_PATH)
    parser.add_argument("--set-pooling-config", type=Path, default=SET_POOLING_CONFIG_PATH)
    parser.add_argument("--encoder-checkpoint", type=Path, default=DEFAULT_ENCODER_CHECKPOINT_PATH)
    parser.add_argument(
        "--no-pretrained-encoder",
        action="store_true",
        help="Skip the Phase 5 pretrained checkpoint entirely and start the PlayerEncoder randomly "
        "initialized (architecture read from --encoder-config instead) -- an ablation to see how much "
        "of GamePredictor's performance comes from pretraining vs. GamePredictor's own training signal.",
    )
    parser.add_argument(
        "--encoder-config",
        type=Path,
        default=PLAYER_ENCODER_CONFIG_PATH,
        help="PlayerEncoder architecture YAML, only used with --no-pretrained-encoder (otherwise the "
        "architecture comes from --encoder-checkpoint itself).",
    )
    parser.add_argument("--pitches-dir", type=Path, default=PROCESSED_DATA_DIR / "pitches")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--games-dir", type=Path, default=GAMES_DIR)
    parser.add_argument("--pitcher-appearances-dir", type=Path, default=PITCHER_APPEARANCES_DIR)
    parser.add_argument("--batter-appearances-dir", type=Path, default=BATTER_APPEARANCES_DIR)
    parser.add_argument("--bullpen-window-days", type=int, default=DEFAULT_BULLPEN_WINDOW_DAYS)
    parser.add_argument("--max-lineup-size", type=int, default=DEFAULT_MAX_LINEUP_SIZE)
    parser.add_argument("--epochs", type=int, default=25, help="Upper bound on epochs; early stopping may end it sooner.")
    parser.add_argument("--batch-size", type=int, default=32)
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--limit-games", type=int, default=None, help="Cap train/val games to the most recent N (smoke-testing only)."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_SEQUENCE_CACHE_DIR),
        help="Disk cache for tokenized player pitch sequences (one file per player), so repeat epochs -- and "
        "later backtest.py runs -- don't rebuild them from scratch. Pass an empty string to disable.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)

    training_config = GamePredictorTrainingConfig.from_yaml(args.training_config)

    logger.info(
        "Season split -- train: %d-%d, val: %s, held out for later testing: %d-%d",
        *TRAIN_SEASON_RANGE, VAL_SEASONS, *TEST_SEASON_RANGE,
    )
    if args.no_pretrained_encoder:
        logger.info(
            "--no-pretrained-encoder set: building a randomly initialized PlayerEncoder from %s "
            "(ignoring --encoder-checkpoint)", args.encoder_config,
        )
        player_encoder, continuous_stats = build_random_encoder(args.encoder_config, args.pitches_dir)
    else:
        logger.info("Loading pretrained PlayerEncoder from %s", args.encoder_checkpoint)
        player_encoder, continuous_stats = load_pretrained_encoder(args.encoder_checkpoint)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    logger.info("Loading game datasets from %s", args.pitches_dir)
    train_dataset, val_dataset, train_games = _load_datasets(
        args.pitches_dir,
        player_encoder.config.max_seq_len,
        args.bullpen_window_days,
        args.max_lineup_size,
        continuous_stats,
        args.raw_dir,
        args.games_dir,
        args.pitcher_appearances_dir,
        args.batter_appearances_dir,
        cache_dir,
    )

    if args.limit_games:
        train_dataset.games = train_dataset.games.tail(args.limit_games).reset_index(drop=True)
        val_dataset.games = val_dataset.games.tail(max(args.limit_games // 5, 1)).reset_index(drop=True)

    logger.info("Train games: %d, Val games: %d", len(train_dataset), len(val_dataset))

    if cache_dir is not None:
        logger.info("Warming player-sequence disk cache at %s (one-time cost per game/date not seen before)", cache_dir)
        warm_start = time.time()
        for name, ds in [("train", train_dataset), ("val", val_dataset)]:
            pitcher_new, batter_new = ds.warm_cache()
            logger.info("  %s: computed %d new pitcher + %d new batter sequences", name, pitcher_new, batter_new)
        logger.info("Cache warm took %.1fs", time.time() - warm_start)

    rest_day_mean, rest_day_std = _compute_rest_day_stats(train_games)
    collate_fn = GameBatchCollator(rest_day_mean, rest_day_std)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers
    )

    loaded_pooling_config = PlayerSetPoolerConfig.from_yaml(args.set_pooling_config)
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

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / "train_game_predictor.csv"
    checkpoint_path = args.checkpoint_dir / "game_predictor_best.pt"

    write_header = not log_path.exists()
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    with open(log_path, "a", newline="") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(
                [
                    "epoch", "encoder_trainable",
                    "train_loss", "train_brier", "train_home_mae", "train_away_mae",
                    "val_loss", "val_brier", "val_home_mae", "val_away_mae",
                ]
            )

        for epoch in range(1, args.epochs + 1):
            encoder_trainable = _encoder_trainable_this_epoch(epoch, training_config)

            train_metrics = run_epoch(
                system, train_loader, device,
                optimizer=optimizer, scaler=scaler, use_amp=use_amp, encoder_trainable=encoder_trainable,
            )
            val_metrics = run_epoch(system, val_loader, device, use_amp=use_amp)

            logger.info(
                "Epoch %d/%d (encoder %s) - train_loss=%.4f train_brier=%.4f train_home_mae=%.4f train_away_mae=%.4f | "
                "val_loss=%.4f val_brier=%.4f val_home_mae=%.4f val_away_mae=%.4f",
                epoch, args.epochs, "trainable" if encoder_trainable else "frozen",
                train_metrics["loss"], train_metrics["brier"], train_metrics["home_mae"], train_metrics["away_mae"],
                val_metrics["loss"], val_metrics["brier"], val_metrics["home_mae"], val_metrics["away_mae"],
            )
            writer.writerow(
                [
                    epoch, encoder_trainable,
                    train_metrics["loss"], train_metrics["brier"], train_metrics["home_mae"], train_metrics["away_mae"],
                    val_metrics["loss"], val_metrics["brier"], val_metrics["home_mae"], val_metrics["away_mae"],
                ]
            )
            log_file.flush()

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
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
                        **{f"val_{k}": v for k, v in val_metrics.items()},
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

    logger.info("Done. Best val_loss=%.4f", best_val_loss)


if __name__ == "__main__":
    main()
