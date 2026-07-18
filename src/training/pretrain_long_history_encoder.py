"""Pretrain LongHistoryEncoder (src/models/long_history_encoder.py) on the
same "predict the next pitch's outcome" task pretrain_encoder.py uses for
the flat PlayerEncoder -- same target, same train/val season split, same
logged metrics -- but drawing on a pitcher's full career-length history
(up to CareerEncoder's max_chunks calendar months) instead of PlayerEncoder's
200-pitch trailing window.

Each pitch is still one training example. Its input history is chunked by
calendar month rather than flattened: up to max_chunks chunks, most recent
last -- the pitcher's most recent *completed* months (however many games
each of those spans), plus (as the final, partial chunk) however many
pitches they've already thrown so far *this month* before the target pitch.
That in-progress prefix is included on purpose, not trimmed off: it's where
most of a next-pitch prediction's real signal already lives (what has this
pitcher shown recently), and dropping it would make this a different, less
comparable task than the original flat version rather than the same task
with a longer reach. Each chunk is itself capped at max_pitch_len
(ChunkEncoder's max_seq_len) pitches, tail-truncated to the most recent --
the same truncation rule NextPitchDataset applies to its one flat window,
just per month here. Chunking by month instead of by game is what makes a
36-chunk cap (roughly three years) practical to encode at all -- the same
cap expressed in games would need ~400 chunks (see git history), and every
chunk-shaped tensor in a batch scales with max_chunks.

A pitcher's first pitch of their career (zero prior chunks) exercises
CareerEncoder's no-history embedding, the career-level equivalent of
PlayerEncoder's own zero-pitch cold start.
"""

from __future__ import annotations

import os

# Must be set before torch's CUDA allocator initializes (i.e. before `import
# torch` triggers any CUDA context creation) -- PYTORCH_CUDA_ALLOC_CONF is
# read once at that point, not re-checked later. expandable_segments lets
# the allocator grow one underlying segment in place instead of carving out
# many fixed-size blocks, which is the standard fix when *reserved* memory
# running well above *allocated* memory points at fragmentation rather than
# genuine peak need (exactly what this training run's memory report showed).
# setdefault, not a plain assignment: respects an explicit override from the
# environment instead of silently clobbering it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler

from src.data.sequence_dataset import (
    CONTINUOUS_FEATURES,
    MATCHUP_INDEX,
    OUTCOME_INDEX,
    OUTCOME_VOCAB,
    PITCH_TYPE_INDEX,
    PlayerPitchSequenceDataset,
    category_indices,
)
from src.data.statcast_common import PROCESSED_DATA_DIR, TRAIN_SEASON_RANGE, VAL_SEASONS
from src.device import DEFAULT_DEVICE, resolve_device
from src.models.long_history_encoder import (
    DEFAULT_CAREER_CONFIG_PATH,
    CareerEncoder,
    CareerEncoderConfig,
    ChunkEncoder,
    ChunkEncoderConfig,
    LongHistoryEncoder,
)
from src.models.player_encoder import DEFAULT_CONFIG_PATH as CHUNK_CONFIG_PATH
from src.resumable_job import write_progress
from src.training.pretrain_encoder import EARLY_STOPPING_PATIENCE, load_season_split, naive_baseline_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PITCHES_DIR = PROCESSED_DATA_DIR / "pitches"
NS_PER_DAY = 86_400_000_000_000

# 2026-07-18: per-epoch training cost on the real full dataset is multiple
# hours (empirically ~2.5-2.9hr on the boundary-1-sized train split), far
# exceeding any single resumable_job.py attempt window (~30-45min, itself
# bounded by this environment's observed ~45-55min background-job ceiling
# -- see src/resumable_job.py's module docstring). Epoch-boundary-only
# resumability (checkpoint once per *completed* epoch, like
# train_event_model.py does) would therefore risk losing hours of GPU
# compute to a single silent background-job kill. RESUME_STATE_FILENAME and
# PROGRESS_FILENAME below back genuine sub-epoch (per-batch) resumability:
# a periodic checkpoint mid-epoch, resumable at the exact batch it left off.
RESUME_STATE_FILENAME = "long_history_encoder_resume_state.pt"
PROGRESS_FILENAME = "long_history_encoder_progress.json"
DEFAULT_CHECKPOINT_INTERVAL_SECONDS = 600.0  # 10min -- comfortably below the attempt window, without excessive torch.save overhead.


class NextPitchLongHistoryDataset(Dataset):
    """One sample per pitch -- see module docstring for the target and how
    its chunked input history is built. Requires `pitches` to already be
    filtered to `is_valid` rows -- see load_season_split (reused as-is from
    pretrain_encoder.py, since the season/validity filtering is unchanged)."""

    def __init__(
        self,
        pitches: pd.DataFrame,
        max_chunks: int,
        max_pitch_len: int,
        continuous_stats: dict[str, tuple[float, float]],
        cache_dir: Path | None = None,
    ) -> None:
        pitches = pitches.sort_values(
            ["pitcher_id", "game_date", "at_bat_number", "pitch_number"]
        ).reset_index(drop=True)

        self.max_chunks = max_chunks
        self.max_pitch_len = max_pitch_len
        # One directory per dataset instance's own (max_chunks, max_pitch_len):
        # cached ranges are only valid for the config that produced them, and
        # mixing configs across a shared cache_dir would silently serve stale
        # truncation/chunk-count decisions. "by_month" tags the chunk-boundary
        # *scheme* itself, so a cache built before the game->month change
        # can never be silently misread as if it were month-chunked.
        self.cache_dir = (
            Path(cache_dir) / f"by_month_chunks{max_chunks}_pitches{max_pitch_len}" if cache_dir is not None else None
        )

        n = len(pitches)
        self.pitcher_id = pitches["pitcher_id"].to_numpy()
        # int64 ns, matching PlayerPitchSequenceDataset's own unit-normalization
        # convention -- game_date's on-disk unit varies (seen both us and ns).
        self.game_date_ns = pitches["game_date"].to_numpy().astype("datetime64[ns]").astype("int64")
        # YYYYMM as a plain int, e.g. 202403 -- monotonically increasing across
        # a year boundary (202312 < 202401), so "did this change" is exactly
        # "did the calendar month change," with no special-casing needed.
        game_date = pitches["game_date"]
        year_month = (game_date.dt.year * 100 + game_date.dt.month).to_numpy()

        cumcount = pitches.groupby("pitcher_id").cumcount().to_numpy()
        self.group_start = np.arange(n) - cumcount

        # A "chunk" is one (pitcher, calendar month) pair -- however many
        # games that month happens to span for this pitcher. Rows are already
        # sorted by pitcher then chronologically, so a chunk boundary is just
        # "the pitcher changed, or the calendar month changed" -- both
        # computed once, up front, rather than per sample.
        new_chunk = np.ones(n, dtype=bool)
        if n > 1:
            new_chunk[1:] = (self.pitcher_id[1:] != self.pitcher_id[:-1]) | (year_month[1:] != year_month[:-1])
        self.chunk_id = np.cumsum(new_chunk) - 1
        self.chunk_start = np.flatnonzero(new_chunk)
        self.chunk_end = np.append(self.chunk_start[1:], n)
        # The chunk id of each row's own pitcher's very first chunk -- bounds
        # how far back __getitem__'s backward walk through chunks can go
        # before it would cross into an earlier, different pitcher.
        self.pitcher_first_chunk = self.chunk_id[self.group_start]

        # Vectorized (no per-sample walk needed): how many chunks -- complete
        # prior months plus, if any pitches this month precede it, one
        # partial current-month chunk -- sample idx would actually see,
        # capped at max_chunks. Used to bucket batches by similar chunk
        # count (see BucketByChunkCountSampler) without paying the full
        # _compute_chunk_ranges walk just to find out how long each sample's
        # history is.
        current_chunk_start_per_row = self.chunk_start[self.chunk_id]
        has_partial = (np.arange(n) > current_chunk_start_per_row).astype(np.int64)
        prior_complete = self.chunk_id - self.pitcher_first_chunk
        self.num_chunks_per_sample = np.minimum(prior_complete + has_partial, max_chunks)

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

        self._loaded_pitchers: set = set()
        # idx -> list of (local_start, end, days_before_cutoff), the resolved
        # (already-truncated) chunk boundaries -- everything __getitem__
        # needs to slice tensors, without re-walking chunk_id/chunk_start.
        self._chunk_range_cache: dict[int, list[tuple[int, int, float]]] = {}

    def __len__(self) -> int:
        return len(self.outcome_idx)

    def _compute_chunk_ranges(self, idx: int) -> list[tuple[int, int, float]]:
        """The expensive part: walk backward from `idx`'s own calendar month
        through this pitcher's prior months until max_chunks is reached or
        this pitcher's history is exhausted. Returns resolved (local_start,
        end, days_before_cutoff) triples, oldest first -- local_start already
        accounts for per-chunk max_pitch_len truncation, so __getitem__ (or
        a cache hit) never needs to redo this walk to use the result."""
        c = int(self.chunk_id[idx])
        current_chunk_start = int(self.chunk_start[c])
        pitcher_first_chunk = int(self.pitcher_first_chunk[idx])

        # Collected most-recent-first (the in-progress current month, if any
        # pitches precede idx within it, then each prior completed month
        # walking backward), reversed to chronological order below.
        chunk_ranges = []
        if idx > current_chunk_start:
            chunk_ranges.append((current_chunk_start, idx))
        next_c = c - 1
        while next_c >= pitcher_first_chunk and len(chunk_ranges) < self.max_chunks:
            chunk_ranges.append((int(self.chunk_start[next_c]), int(self.chunk_end[next_c])))
            next_c -= 1
        chunk_ranges.reverse()

        target_date_ns = self.game_date_ns[idx]
        resolved = []
        for start, end in chunk_ranges:
            length = min(end - start, self.max_pitch_len)
            local_start = end - length  # tail-truncate: keep this chunk's most recent pitches
            # A month-chunk spans many calendar dates -- "recency" should reflect
            # the chunk's most recent activity (its last row), not the date its
            # month happened to start on.
            days_before_cutoff = float((target_date_ns - self.game_date_ns[end - 1]) / NS_PER_DAY)
            resolved.append((local_start, end, days_before_cutoff))
        return resolved

    def _cache_path(self, pitcher_id: int) -> Path:
        return self.cache_dir / f"{pitcher_id}.pt"

    def _ensure_pitcher_cache_loaded(self, pitcher_id: int) -> None:
        if pitcher_id in self._loaded_pitchers:
            return
        self._loaded_pitchers.add(pitcher_id)
        path = self._cache_path(pitcher_id)
        if path.exists():
            self._chunk_range_cache.update(torch.load(path, weights_only=False))

    def _get_chunk_ranges(self, idx: int) -> list[tuple[int, int, float]]:
        if self.cache_dir is not None:
            self._ensure_pitcher_cache_loaded(int(self.pitcher_id[idx]))
            cached = self._chunk_range_cache.get(idx)
            if cached is not None:
                return cached
        return self._compute_chunk_ranges(idx)

    def precompute_and_cache(self) -> int:
        """Computes every sample's chunk ranges not already cached and
        writes them to disk, one file per pitcher (merged with whatever
        that pitcher's file already held, so repeated warm calls accumulate
        rather than clobber) -- same convention as
        PlayerPitchSequenceDataset.precompute_and_cache. Call this once,
        single-process, before training starts; __getitem__ itself never
        writes, which is what makes it safe to read this same cache_dir
        from multiple DataLoader worker processes afterwards.

        Returns the number of samples actually computed (as opposed to
        already-cached).
        """
        if self.cache_dir is None:
            raise ValueError("cache_dir was not set on this dataset")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        dirty_pitchers: dict[int, dict[int, list]] = {}
        computed = 0
        for idx in range(len(self)):
            pitcher_id = int(self.pitcher_id[idx])
            self._ensure_pitcher_cache_loaded(pitcher_id)
            if idx in self._chunk_range_cache:
                continue
            ranges = self._compute_chunk_ranges(idx)
            self._chunk_range_cache[idx] = ranges
            dirty_pitchers.setdefault(pitcher_id, {})[idx] = ranges
            computed += 1

        for pitcher_id, entries in dirty_pitchers.items():
            path = self._cache_path(pitcher_id)
            existing = torch.load(path, weights_only=False) if path.exists() else {}
            existing.update(entries)
            torch.save(existing, path)

        return computed

    def __getitem__(self, idx: int) -> dict:
        target = int(self.outcome_idx[idx])
        resolved = self._get_chunk_ranges(idx)

        if not resolved:
            return {"has_history": False, "num_chunks": 0, "chunks": [], "target": target}

        chunks = []
        for local_start, end, days_before_cutoff in resolved:
            chunks.append(
                {
                    "length": end - local_start,
                    "continuous": torch.from_numpy(self.continuous[local_start:end]),
                    "pitch_type": torch.from_numpy(self.pitch_type_idx[local_start:end]),
                    "outcome": torch.from_numpy(self.outcome_idx[local_start:end]),
                    "matchup": torch.from_numpy(self.matchup_idx[local_start:end]),
                    "days_before_cutoff": days_before_cutoff,
                }
            )

        return {"has_history": True, "num_chunks": len(chunks), "chunks": chunks, "target": target}


class BucketByChunkCountSampler(Sampler[list[int]]):
    """Batches samples so players with similar total chunk counts land
    together, instead of shuffle=True's plain random grouping -- one
    36-chunk veteran drawn into an otherwise short-history batch forces
    every other sample in that batch to pad up to 36 chunks, wasting most
    of the padded tensor. Sorting by num_chunks first (stable, so ties keep
    their original relative order -- deterministic batch membership run to
    run) means adjacent samples in the sort are close in chunk count, so
    each fixed-size batch's padding waste is close to that batch's own
    worst case, not the whole dataset's.

    Batch *membership* is fixed by the sort and never reshuffled -- only
    the order batches are yielded in is shuffled, once per __iter__ call
    (DataLoader calls this fresh every epoch, so that's still a different
    batch order each epoch). Not shuffling within a batch or adding
    cross-batch noise: the point is to bucket by length, and additional
    randomness there would just re-introduce the padding waste this exists
    to avoid.
    """

    def __init__(self, num_chunks_per_sample: np.ndarray, batch_size: int, shuffle: bool = True, seed: int | None = None) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        order = np.argsort(num_chunks_per_sample, kind="stable")
        self.batches = [order[i : i + batch_size].tolist() for i in range(0, len(order), batch_size)]
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        batch_order = list(range(len(self.batches)))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self._epoch if self.seed is not None else None)
            rng.shuffle(batch_order)
            self._epoch += 1
        for i in batch_order:
            yield self.batches[i]


def batch_order_for_epoch(sampler: BucketByChunkCountSampler, epoch: int) -> list[int]:
    """The exact shuffled batch order BucketByChunkCountSampler.__iter__
    would yield on its `epoch`-th call (1-indexed here; the sampler's own
    internal `_epoch` counter is 0-indexed and only advances via real
    __iter__ calls), computed directly from `sampler.seed` instead of by
    calling __iter__ repeatedly. This is what makes sub-epoch resumability
    possible: a resumed run can reconstruct any epoch's batch order (to
    skip the batches it already finished before an interruption) without
    depending on the sampler's own mutable, call-count-based state.
    Requires a concrete `sampler.seed` (not None) -- resumability is only
    meaningful if the batch order is reproducible."""
    if sampler.seed is None:
        raise ValueError("batch_order_for_epoch requires a concrete sampler.seed for reproducibility.")
    batch_order = list(range(len(sampler.batches)))
    if sampler.shuffle:
        rng = np.random.default_rng(sampler.seed + (epoch - 1))
        rng.shuffle(batch_order)
    return batch_order


def collate_long_history_batch(
    batch: list[dict],
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads to this batch's own longest chunk count and longest per-chunk
    pitch count -- both vary sample to sample, same "pad to the batch max"
    convention collate_next_pitch_batch uses for the flat pitch dimension.

    Returns (chunk_pitch_sequences, days_before_cutoff, chunk_padding_mask,
    has_history, targets) -- the first four are exactly LongHistoryEncoder.
    forward's positional arguments.
    """
    batch_size = len(batch)
    max_chunks = max(max((sample["num_chunks"] for sample in batch), default=0), 1)
    max_pitch_len = max(
        max((chunk["length"] for sample in batch for chunk in sample["chunks"]), default=0), 1
    )

    n_features = len(CONTINUOUS_FEATURES)
    continuous = torch.zeros(batch_size, max_chunks, max_pitch_len, n_features)
    pitch_type = torch.zeros(batch_size, max_chunks, max_pitch_len, dtype=torch.long)
    outcome = torch.zeros(batch_size, max_chunks, max_pitch_len, dtype=torch.long)
    matchup = torch.zeros(batch_size, max_chunks, max_pitch_len, dtype=torch.long)
    position = torch.zeros(batch_size, max_chunks, max_pitch_len, dtype=torch.long)
    pitch_padding_mask = torch.ones(batch_size, max_chunks, max_pitch_len, dtype=torch.bool)
    chunk_has_history = torch.zeros(batch_size, max_chunks, dtype=torch.bool)

    days_before_cutoff = torch.zeros(batch_size, max_chunks)
    chunk_padding_mask = torch.ones(batch_size, max_chunks, dtype=torch.bool)
    has_history = torch.zeros(batch_size, dtype=torch.bool)
    targets = torch.zeros(batch_size, dtype=torch.long)

    for i, sample in enumerate(batch):
        targets[i] = sample["target"]
        has_history[i] = sample["has_history"]
        num_chunks = sample["num_chunks"]
        if num_chunks == 0:
            continue
        chunk_padding_mask[i, :num_chunks] = False
        for c, chunk in enumerate(sample["chunks"]):
            length = chunk["length"]
            chunk_has_history[i, c] = True
            days_before_cutoff[i, c] = chunk["days_before_cutoff"]
            continuous[i, c, :length] = chunk["continuous"]
            pitch_type[i, c, :length] = chunk["pitch_type"]
            outcome[i, c, :length] = chunk["outcome"]
            matchup[i, c, :length] = chunk["matchup"]
            position[i, c, :length] = torch.arange(length)
            pitch_padding_mask[i, c, :length] = False

    chunk_pitch_sequences = {
        "continuous": continuous,
        "pitch_type": pitch_type,
        "outcome": outcome,
        "matchup": matchup,
        "position": position,
        "padding_mask": pitch_padding_mask,
        "has_history": chunk_has_history,
    }
    return chunk_pitch_sequences, days_before_cutoff, chunk_padding_mask, has_history, targets


class NextPitchLongHistoryPredictor(nn.Module):
    def __init__(self, chunk_config: ChunkEncoderConfig, career_config: CareerEncoderConfig) -> None:
        super().__init__()
        self.encoder = LongHistoryEncoder(ChunkEncoder(chunk_config), CareerEncoder(career_config))
        self.classifier = nn.Linear(career_config.hidden_size, len(OUTCOME_VOCAB))

    def forward(
        self,
        chunk_pitch_sequences: dict[str, torch.Tensor],
        days_before_cutoff: torch.Tensor,
        chunk_padding_mask: torch.Tensor,
        has_history: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.encoder(chunk_pitch_sequences, days_before_cutoff, chunk_padding_mask, has_history)
        return self.classifier(embedding)


def run_epoch(
    model: NextPitchLongHistoryPredictor,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
    log_every: int | None = None,
) -> tuple[float, float]:
    train = optimizer is not None
    model.train(mode=train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    num_batches = len(loader)
    epoch_start = time.time()

    with torch.set_grad_enabled(train):
        for batch_idx, (chunk_pitch_sequences, days_before_cutoff, chunk_padding_mask, has_history, targets) in enumerate(loader, start=1):
            # Each batch's own max_chunks/max_pitch_len (set by whichever samples
            # landed in it) can vary a lot -- log per-batch shape and throughput
            # so a slow run is diagnosable instead of an opaque total-time wait.
            if log_every and (batch_idx % log_every == 0 or batch_idx == num_batches):
                elapsed = time.time() - epoch_start
                batches_per_sec = batch_idx / elapsed if elapsed > 0 else float("inf")
                eta = (num_batches - batch_idx) / batches_per_sec if batches_per_sec > 0 else float("inf")
                _, max_chunks, max_pitch_len, _ = chunk_pitch_sequences["continuous"].shape
                logger.info(
                    "  batch %d/%d (%.1fs elapsed, %.2f batches/s, ETA %.0fs) -- this batch's shape: "
                    "chunks=%d pitch_len=%d",
                    batch_idx, num_batches, elapsed, batches_per_sec, eta, max_chunks, max_pitch_len,
                )

            chunk_pitch_sequences = {k: v.to(device) for k, v in chunk_pitch_sequences.items()}
            days_before_cutoff = days_before_cutoff.to(device)
            chunk_padding_mask = chunk_padding_mask.to(device)
            has_history = has_history.to(device)
            targets = targets.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(chunk_pitch_sequences, days_before_cutoff, chunk_padding_mask, has_history)
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


def run_train_epoch_resumable(
    model: NextPitchLongHistoryPredictor,
    dataset: "NextPitchLongHistoryDataset",
    sampler: BucketByChunkCountSampler,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    use_amp: bool,
    epoch: int,
    start_batch_index: int,
    start_loss_sum: float,
    start_correct_sum: int,
    start_count_sum: int,
    checkpoint_interval_seconds: float,
    on_checkpoint,
    log_every: int | None = None,
) -> tuple[float, float]:
    """Train-pass counterpart to run_epoch, resumable at sub-epoch (batch)
    granularity: walks a precomputed batch order (batch_order_for_epoch)
    directly instead of iterating a DataLoader, so a resumed run can start
    at `start_batch_index` without redoing already-completed batches and
    without needing to persist/restore DataLoader-internal iterator state
    (which PyTorch doesn't expose for exactly this purpose). Equivalent to
    DataLoader iteration with num_workers=0 (this script's own default),
    just with the starting position under explicit control.

    start_loss_sum/start_correct_sum/start_count_sum: the train-pass
    accumulators to resume from (0/0/0 for a fresh epoch, or whatever a
    prior interrupted attempt had accumulated through start_batch_index).

    on_checkpoint(batch_index, loss_sum, correct_sum, count_sum) is called
    every checkpoint_interval_seconds of wall-clock, and once more
    unconditionally after the final batch -- the caller decides what
    "checkpoint" means (saving model/optimizer/scaler state plus these
    accumulators, and updating the resumable_job.py progress file)."""
    model.train()
    batch_order = batch_order_for_epoch(sampler, epoch)
    num_batches = len(batch_order)

    loss_sum, correct_sum, count_sum = start_loss_sum, start_correct_sum, start_count_sum
    last_checkpoint = time.time()
    epoch_start = time.time()

    for progress_idx, batch_pos in enumerate(range(start_batch_index, num_batches), start=1):
        indices = sampler.batches[batch_order[batch_pos]]
        batch = [dataset[i] for i in indices]
        chunk_pitch_sequences, days_before_cutoff, chunk_padding_mask, has_history, targets = collate_long_history_batch(batch)

        if log_every and (progress_idx % log_every == 0 or batch_pos == num_batches - 1):
            elapsed = time.time() - epoch_start
            batches_per_sec = progress_idx / elapsed if elapsed > 0 else float("inf")
            eta = (num_batches - batch_pos - 1) / batches_per_sec if batches_per_sec > 0 else float("inf")
            _, max_chunks, max_pitch_len, _ = chunk_pitch_sequences["continuous"].shape
            logger.info(
                "  batch %d/%d (%.1fs elapsed, %.2f batches/s, ETA %.0fs) -- this batch's shape: "
                "chunks=%d pitch_len=%d",
                batch_pos + 1, num_batches, elapsed, batches_per_sec, eta, max_chunks, max_pitch_len,
            )

        chunk_pitch_sequences = {k: v.to(device) for k, v in chunk_pitch_sequences.items()}
        days_before_cutoff = days_before_cutoff.to(device)
        chunk_padding_mask = chunk_padding_mask.to(device)
        has_history = has_history.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(chunk_pitch_sequences, days_before_cutoff, chunk_padding_mask, has_history)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        correct_sum += (logits.argmax(dim=-1) == targets).sum().item()
        count_sum += batch_size

        if time.time() - last_checkpoint >= checkpoint_interval_seconds:
            on_checkpoint(batch_pos + 1, loss_sum, correct_sum, count_sum)
            last_checkpoint = time.time()

    on_checkpoint(num_batches, loss_sum, correct_sum, count_sum)
    return loss_sum / count_sum, correct_sum / count_sum


def _sample_by_pitcher(pitches: pd.DataFrame, frac: float, seed: int | None) -> pd.DataFrame:
    """Randomly keeps `frac` of the distinct pitchers in `pitches`, with all
    of each kept pitcher's rows intact -- see the call site for why this
    can't be a plain row-level .sample()."""
    pitcher_ids = pitches["pitcher_id"].unique()
    rng = np.random.default_rng(seed)
    n_keep = max(1, round(len(pitcher_ids) * frac))
    kept = rng.choice(pitcher_ids, size=n_keep, replace=False)
    return pitches[pitches["pitcher_id"].isin(kept)].reset_index(drop=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain LongHistoryEncoder (ChunkEncoder + CareerEncoder) on next-pitch-outcome "
        "prediction, using each pitcher's full career-length history instead of a flat 200-pitch window."
    )
    parser.add_argument("--chunk-config", type=Path, default=CHUNK_CONFIG_PATH, help="ChunkEncoder YAML config.")
    parser.add_argument("--career-config", type=Path, default=DEFAULT_CAREER_CONFIG_PATH, help="CareerEncoder YAML config.")
    parser.add_argument("--pitches-dir", type=Path, default=DEFAULT_PITCHES_DIR)
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Smaller default than pretrain_encoder.py's 256 -- each sample here can carry up to "
        "max_chunks*max_pitch_len pitch-feature slots instead of max_seq_len, a much larger per-sample footprint.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Stop if validation loss doesn't improve for this many consecutive epochs.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--log-every-n-batches",
        type=int,
        default=None,
        help="Log per-batch progress (elapsed time, batches/s, ETA, this batch's chunk/pitch-len shape) "
        "every N batches. Off by default; each batch's cost varies a lot with which samples land in it, "
        "so this is useful for diagnosing a slow run rather than waiting on an opaque total.",
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
        "--limit-rows",
        type=int,
        default=None,
        help="Cap train/val rows to the most recent N (smoke-testing only).",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Randomly subsample train/val to this fraction (e.g. 0.05 for 5%%) -- for timing/memory "
        "dry runs against a representative (not just most-recent-N) slice of the real data. Applied "
        "after --limit-rows if both are given.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for --sample-frac, and (unconditionally, regardless of whether --sample-frac is "
        "used) for the train batch sampler's per-epoch shuffle order -- a concrete seed here is what makes "
        "sub-epoch resumability possible (batch_order_for_epoch must be able to reproduce any epoch's exact "
        "batch order to resume mid-epoch). Defaults to 0, not None, for that reason: an unset seed would "
        "still train fine but couldn't be resumed deterministically.",
    )
    parser.add_argument(
        "--checkpoint-interval-seconds",
        type=float,
        default=DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
        help="How often (wall-clock, during the train pass) to save a resumable checkpoint of full training "
        "state (model/optimizer/scaler + exact batch position) and update the resumable_job.py progress "
        "file. Should be comfortably below whatever --attempt-timeout-seconds run_until_complete is using.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Disk cache for each sample's resolved chunk boundaries (one file per pitcher), so repeated "
        "epochs skip re-walking a pitcher's month-by-month history from scratch -- NextPitchLongHistoryDataset "
        "recomputes it fresh every access otherwise. Off by default; pass a directory to enable it. Note "
        "this only removes the CPU-side bookkeeping cost, not the GPU compute of re-encoding a deep "
        "pitcher's real months through the (still-training) chunk encoder every epoch.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)  # reproducible model weight init, matching train_event_model.py's own convention
    device = resolve_device(args.device)
    chunk_config = ChunkEncoderConfig.from_yaml(args.chunk_config)
    career_config = CareerEncoderConfig.from_yaml(args.career_config)

    train_season_range = (args.train_season_start, args.train_season_end)
    val_seasons = tuple(args.val_seasons)
    logger.info("Season split -- train: %d-%d, val: %s", *train_season_range, val_seasons)
    logger.info("Loading processed pitches from %s", args.pitches_dir)
    train_df, val_df = load_season_split(args.pitches_dir, train_season_range, val_seasons)

    if args.limit_rows:
        train_df = train_df.tail(args.limit_rows).reset_index(drop=True)
        val_df = val_df.tail(max(args.limit_rows // 5, 1)).reset_index(drop=True)

    if args.sample_frac:
        # Sample whole pitchers, not individual pitch rows: NextPitchLongHistoryDataset's
        # chunk boundaries depend on a pitcher's own history being intact and
        # contiguous, so randomly dropping individual pitches would fragment
        # games and understate real chunk depth/cost -- sampling pitcher_ids
        # keeps every included pitcher's full chronological history whole,
        # which is what makes the resulting subset actually representative.
        train_df = _sample_by_pitcher(train_df, args.sample_frac, args.seed)
        val_df = _sample_by_pitcher(val_df, args.sample_frac, args.seed)
        logger.info(
            "--sample-frac=%.4f: randomly sampled that fraction of pitchers (with their full history intact)",
            args.sample_frac,
        )

    logger.info("Train pitches: %d, Val pitches: %d", len(train_df), len(val_df))

    baseline_accuracy, baseline_loss, majority_class = naive_baseline_metrics(val_df)
    logger.info(
        "Naive baseline (always predict %r) -- val_loss=%.4f val_accuracy=%.4f",
        majority_class,
        baseline_loss,
        baseline_accuracy,
    )

    continuous_stats = PlayerPitchSequenceDataset._compute_continuous_stats(train_df)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    train_dataset = NextPitchLongHistoryDataset(
        train_df, career_config.max_chunks, chunk_config.max_seq_len, continuous_stats,
        cache_dir=(cache_dir / "train") if cache_dir else None,
    )
    val_dataset = NextPitchLongHistoryDataset(
        val_df, career_config.max_chunks, chunk_config.max_seq_len, continuous_stats,
        cache_dir=(cache_dir / "val") if cache_dir else None,
    )

    if cache_dir is not None:
        for name, dataset in [("train", train_dataset), ("val", val_dataset)]:
            warm_start = time.time()
            computed = dataset.precompute_and_cache()
            logger.info(
                "%s chunk-range cache: computed %d new sample(s) in %.1fs (cache_dir=%s)",
                name, computed, time.time() - warm_start, dataset.cache_dir,
            )

    train_sampler = BucketByChunkCountSampler(
        train_dataset.num_chunks_per_sample, args.batch_size, shuffle=True, seed=args.seed
    )
    val_sampler = BucketByChunkCountSampler(
        val_dataset.num_chunks_per_sample, args.batch_size, shuffle=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_long_history_batch,
        num_workers=args.num_workers,
    )
    batches_per_epoch = len(train_sampler)

    model = NextPitchLongHistoryPredictor(chunk_config, career_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / "pretrain_long_history_encoder.csv"
    checkpoint_path = args.checkpoint_dir / "long_history_encoder_best.pt"
    resume_state_path = args.checkpoint_dir / RESUME_STATE_FILENAME
    progress_path = args.checkpoint_dir / PROGRESS_FILENAME

    write_header = not log_path.exists()
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_without_improvement = 0
    start_epoch = 1
    batch_index_in_epoch = 0
    epoch_loss_sum, epoch_correct_sum, epoch_count_sum = 0.0, 0, 0
    train_pass_complete_for_current_epoch = False

    # Sub-epoch resumability: a killed/interrupted run picks back up from its
    # own last periodic mid-epoch checkpoint (see run_train_epoch_resumable's
    # on_checkpoint), not just the last *completed* epoch -- per-epoch cost on
    # the real dataset is multiple hours, far exceeding a single
    # resumable_job.py attempt window, so epoch-boundary-only resumability
    # (train_event_model.py's pattern) would risk losing hours of GPU compute
    # to one background-job kill.
    if resume_state_path.exists():
        existing = torch.load(resume_state_path, map_location=device, weights_only=False)
        compatible = (
            existing.get("chunk_config") == asdict(chunk_config)
            and existing.get("career_config") == asdict(career_config)
            and existing.get("sampler_seed") == args.seed
        )
        if not compatible:
            logger.warning(
                "Found a resume-state checkpoint at %s, but its config and/or sampler seed don't match this "
                "run's -- treating it as stale (from a differently-configured run) rather than resuming from "
                "it. Starting fresh.",
                resume_state_path,
            )
        else:
            model.load_state_dict(existing["model_state_dict"])
            optimizer.load_state_dict(existing["optimizer_state_dict"])
            scaler.load_state_dict(existing["scaler_state_dict"])
            start_epoch = existing["epoch"]
            batch_index_in_epoch = existing["batch_index_in_epoch"]
            epoch_loss_sum = existing["epoch_loss_sum"]
            epoch_correct_sum = existing["epoch_correct_sum"]
            epoch_count_sum = existing["epoch_count_sum"]
            train_pass_complete_for_current_epoch = existing["train_pass_complete_for_current_epoch"]
            best_val_loss = existing["best_val_loss"]
            best_val_acc = existing["best_val_acc"]
            epochs_without_improvement = existing["epochs_without_improvement"]
            logger.info(
                "Resuming from %s: epoch %d, batch %d/%d of that epoch's train pass (train_pass_complete=%s).",
                resume_state_path, start_epoch, batch_index_in_epoch, batches_per_epoch, train_pass_complete_for_current_epoch,
            )

    def save_resume_state(epoch: int, batch_idx: int, loss_sum: float, correct_sum: int, count_sum: int, train_complete: bool) -> None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "chunk_config": asdict(chunk_config),
                "career_config": asdict(career_config),
                "continuous_stats": continuous_stats,
                "sampler_seed": args.seed,
                "epoch": epoch,
                "batch_index_in_epoch": batch_idx,
                "epoch_loss_sum": loss_sum,
                "epoch_correct_sum": correct_sum,
                "epoch_count_sum": count_sum,
                "train_pass_complete_for_current_epoch": train_complete,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "epochs_without_improvement": epochs_without_improvement,
            },
            resume_state_path,
        )
        completed_batches = (epoch - 1) * batches_per_epoch + batch_idx
        write_progress(
            progress_path,
            total=args.epochs * batches_per_epoch,
            completed=completed_batches,
            extra={"epoch": epoch, "batch_index_in_epoch": batch_idx, "batches_per_epoch": batches_per_epoch},
        )

    with open(log_path, "a", newline="") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])

        final_completed_batches = (start_epoch - 1) * batches_per_epoch + batch_index_in_epoch
        for epoch in range(start_epoch, args.epochs + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            resuming_this_epoch = epoch == start_epoch
            if resuming_this_epoch and train_pass_complete_for_current_epoch:
                # This attempt's very first epoch had already fully finished its train pass
                # before an earlier interruption -- nothing left to train, go straight to val.
                train_loss = epoch_loss_sum / epoch_count_sum
                train_acc = epoch_correct_sum / epoch_count_sum
            else:
                this_start_batch = batch_index_in_epoch if resuming_this_epoch else 0
                this_start_loss = epoch_loss_sum if resuming_this_epoch else 0.0
                this_start_correct = epoch_correct_sum if resuming_this_epoch else 0
                this_start_count = epoch_count_sum if resuming_this_epoch else 0

                train_start = time.time()
                train_loss, train_acc = run_train_epoch_resumable(
                    model, train_dataset, train_sampler, device, criterion, optimizer, scaler, use_amp,
                    epoch, this_start_batch, this_start_loss, this_start_correct, this_start_count,
                    args.checkpoint_interval_seconds,
                    on_checkpoint=lambda b, l, c, n, _epoch=epoch: save_resume_state(
                        _epoch, b, l, c, n, train_complete=(b >= batches_per_epoch)
                    ),
                    log_every=args.log_every_n_batches,
                )
                train_seconds = time.time() - train_start
                logger.info("Epoch %d train pass wall-clock (this attempt): %.1fs", epoch, train_seconds)

            if device.type == "cuda":
                peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
                peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
                logger.info(
                    "Epoch %d peak GPU memory: %.2f GiB allocated, %.2f GiB reserved", epoch, peak_allocated, peak_reserved
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
                        "chunk_encoder_state_dict": model.encoder.chunk_encoder.state_dict(),
                        "career_encoder_state_dict": model.encoder.career_encoder.state_dict(),
                        "classifier_state_dict": model.classifier.state_dict(),
                        "chunk_config": asdict(chunk_config),
                        "career_config": asdict(career_config),
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

            # Next epoch (if any) starts its train pass fresh regardless of
            # this epoch's outcome -- reset the resumable-state accumulators.
            batch_index_in_epoch = 0
            epoch_loss_sum, epoch_correct_sum, epoch_count_sum = 0.0, 0, 0
            final_completed_batches = epoch * batches_per_epoch

            if epochs_without_improvement >= args.patience:
                logger.info(
                    "Early stopping: val_loss hasn't improved for %d consecutive epochs.",
                    epochs_without_improvement,
                )
                break

    # Mark the job done for resumable_job.py regardless of whether early
    # stopping cut it short of args.epochs -- total is corrected down to
    # whatever was actually completed, so remaining is guaranteed 0 here.
    write_progress(
        progress_path, total=final_completed_batches, completed=final_completed_batches,
        extra={"status": "done", "best_val_loss": best_val_loss, "best_val_accuracy": best_val_acc},
    )

    logger.info(
        "Done. Model val_loss=%.4f val_accuracy=%.4f | Naive baseline val_loss=%.4f val_accuracy=%.4f",
        best_val_loss,
        best_val_acc,
        baseline_loss,
        baseline_accuracy,
    )


if __name__ == "__main__":
    main()
