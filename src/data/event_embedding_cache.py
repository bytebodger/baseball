"""Precomputes and on-disk caches the frozen LongHistoryEncoder's embedding
for every distinct (player_id, game_date) pair the event model needs, for
both the pitcher and batter perspectives -- so the real event-model training
loop never runs the encoder live. A benchmark (see git history /
scratch/benchmark_event_model_precompute.py) measured this forward pass at
~40 minutes for the full 2015-2026 dataset's 804,536 distinct pairs, versus
~2.7 minutes/epoch for the lightweight event head alone; recomputing the
encoder pass every epoch would make it the dominant cost by nearly an order
of magnitude, so it's precomputed once here instead.

On-disk format: one small file per (player_id, game_date) entry, under
cache_dir/{perspective}/{player_id}/{game_date.value (ns int)}.pt, containing
just that entry's embedding tensor. An earlier version instead kept one file
per player -- cache_dir/{perspective}/{player_id}.pt, a dict of
{date_ns: embedding} -- merged and fully re-serialized on every touch. That
degraded badly in practice: write time is dominated by re-pickling the
player's *entire* accumulated dict on every single new date, so it grew with
career length instead of staying flat, and was measured climbing from
~324ms/batch to over 700ms/batch (versus a ~100ms forward pass) as batches
reached players with larger histories -- writes, not the GPU, had become the
bottleneck. Per-entry files make each write's cost independent of how much
else is already cached for that player.

The ~96% of pairs cached under the old per-player-dict format before this
change were not migrated -- they're complete and are never written to again,
so there was nothing to gain from rewriting them, only downtime. Both
formats are read: EmbeddingCache.get() checks the new per-entry file first,
then falls back to that player's old-format dict file if present.
precompute_and_cache_embeddings still reads (but never rewrites) old-format
files too, purely to know what's already cached and skip recomputing it.

Both formats key resumability the same way: re-running
precompute_and_cache_embeddings after adding new seasons only computes
pairs missing from *either* format, and an interrupted run resumes instead
of restarting.

Unlike PlayerPitchSequenceDataset.build_sequence, EmbeddingCache.get never
falls back to computing on a miss -- it raises. A silent live fallback here
would defeat the entire point of precomputing (the whole ~40 minute cost,
one query at a time, invisibly reintroduced into the training loop) and
would mask real cache-build gaps instead of surfacing them. Also unlike that
cache, this one has no meaning independent of the specific encoder
checkpoint it was built from -- retraining the encoder invalidates every
cached embedding, which is on the caller to track by pointing at a fresh
cache_dir (see build_cache_dir_name).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.sequence_dataset import (
    CONTINUOUS_FEATURES,
    MATCHUP_INDEX,
    OUTCOME_INDEX,
    PITCH_TYPE_INDEX,
    category_indices,
)
from src.data.statcast_common import PROCESSED_DATA_DIR, TEST_SEASON_RANGE, TRAIN_SEASON_RANGE, VAL_SEASONS, read_partitioned
from src.device import DEFAULT_DEVICE, resolve_device
from src.models.long_history_encoder import (
    CareerEncoder,
    CareerEncoderConfig,
    ChunkEncoder,
    ChunkEncoderConfig,
    LongHistoryEncoder,
)
from src.training.pretrain_long_history_encoder import (
    NS_PER_DAY,
    BucketByChunkCountSampler,
    collate_long_history_batch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = PROCESSED_DATA_DIR / "event_embedding_cache"
DEFAULT_CHECKPOINT_PATH = Path("checkpoints") / "long_history_encoder_best.pt"
PERSPECTIVES = ("pitcher", "batter")


# ---------------------------------------------------------------------------
# Query-indexed chunk builder: arbitrary (player_id, cutoff_date) -> that
# player's chunked history, for either perspective. NextPitchLongHistoryDataset
# (pretrain_long_history_encoder.py) can't be reused directly here since it's
# pitcher-only and indexed by an existing pitch row (one sample per pitch,
# cutoff = that row's own date) rather than by an arbitrary query pair --
# exactly what precomputing "one embedding per distinct (player, date), not
# per pitch" needs instead.
# ---------------------------------------------------------------------------


def build_chunk_index(pitches: pd.DataFrame, id_column: str) -> dict:
    """Same chunk-boundary bookkeeping NextPitchLongHistoryDataset builds
    (chunk_id/chunk_start/chunk_end/first-chunk-per-player), generalized to
    an arbitrary id_column ("pitcher_id" or "batter_id") and independently
    sorted for that perspective."""
    sorted_pitches = pitches.sort_values(
        [id_column, "game_date", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)
    n = len(sorted_pitches)
    player_id = sorted_pitches[id_column].to_numpy()
    game_date = sorted_pitches["game_date"]
    game_date_ns = game_date.to_numpy().astype("datetime64[ns]").astype("int64")
    year_month = (game_date.dt.year * 100 + game_date.dt.month).to_numpy()

    cumcount = sorted_pitches.groupby(id_column).cumcount().to_numpy()
    group_start = np.arange(n) - cumcount

    new_chunk = np.ones(n, dtype=bool)
    if n > 1:
        new_chunk[1:] = (player_id[1:] != player_id[:-1]) | (year_month[1:] != year_month[:-1])
    chunk_id = np.cumsum(new_chunk) - 1
    chunk_start = np.flatnonzero(new_chunk)
    chunk_end = np.append(chunk_start[1:], n)
    player_first_chunk = chunk_id[group_start]

    continuous_stats = {}
    for col in CONTINUOUS_FEATURES:
        values = sorted_pitches[col].to_numpy(dtype="float64", na_value=np.nan)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        continuous_stats[col] = (mean, std if std > 0 else 1.0)
    continuous = np.stack(
        [(sorted_pitches[c].to_numpy(dtype="float64", na_value=np.nan) - m) / s for c, (m, s) in continuous_stats.items()],
        axis=1,
    )
    continuous = np.nan_to_num(continuous, nan=0.0).astype("float32")
    pitch_type_idx = category_indices(sorted_pitches["pitch_type"], PITCH_TYPE_INDEX).numpy()
    outcome_idx = category_indices(sorted_pitches["outcome"], OUTCOME_INDEX).numpy()
    matchup = sorted_pitches["stand"].astype(object) + "_" + sorted_pitches["p_throws"].astype(object)
    matchup_idx = category_indices(matchup, MATCHUP_INDEX).numpy()

    boundaries = np.flatnonzero(player_id[1:] != player_id[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))
    player_ranges = dict(zip(player_id[starts].tolist(), zip(starts.tolist(), ends.tolist())))

    return {
        "game_date_ns": game_date_ns,
        "chunk_id": chunk_id,
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
        "player_first_chunk": player_first_chunk,
        "player_ranges": player_ranges,
        "continuous": continuous,
        "pitch_type_idx": pitch_type_idx,
        "outcome_idx": outcome_idx,
        "matchup_idx": matchup_idx,
    }


def _local_end_for_query(index: dict, player_id_value, cutoff_date_ns: int) -> tuple[int, int] | None:
    """Returns (start, local_end) -- this player's row range and how much of
    it (exclusive) is strictly before cutoff_date_ns -- or None if the
    player is unknown or has no history before the cutoff."""
    range_ = index["player_ranges"].get(player_id_value)
    if range_ is None:
        return None
    start, end = range_
    preceding = int(np.searchsorted(index["game_date_ns"][start:end], cutoff_date_ns, side="left"))
    if preceding <= 0:
        return None
    return start, start + preceding


def estimate_num_chunks(index: dict, queries: list[tuple[int, int]], max_chunks: int) -> np.ndarray:
    """Cheap (no chunk-range list construction) num_chunks-per-query count,
    for BucketByChunkCountSampler -- same purpose as
    NextPitchLongHistoryDataset.num_chunks_per_sample, just for arbitrary
    query pairs instead of existing rows."""
    counts = np.zeros(len(queries), dtype=np.int64)
    for i, (player_id_value, cutoff_ns) in enumerate(queries):
        located = _local_end_for_query(index, player_id_value, cutoff_ns)
        if located is None:
            continue
        _, local_end = located
        c = int(index["chunk_id"][local_end - 1])
        current_chunk_start = int(index["chunk_start"][c])
        player_first_chunk = int(index["player_first_chunk"][local_end - 1])
        has_partial = 1 if local_end > current_chunk_start else 0
        prior_complete = c - player_first_chunk
        counts[i] = min(prior_complete + has_partial, max_chunks)
    return counts


def chunk_ranges_for_query(
    index: dict, player_id_value, cutoff_date_ns: int, max_chunks: int, max_pitch_len: int
) -> list[tuple[int, int, float]]:
    """Resolved (local_start, end, days_before_cutoff) triples for this
    query, oldest first -- same shape NextPitchLongHistoryDataset._compute_chunk_ranges
    returns, walked backward from an arbitrary cutoff date instead of an
    existing row's own."""
    located = _local_end_for_query(index, player_id_value, cutoff_date_ns)
    if located is None:
        return []
    _, local_end = located

    c = int(index["chunk_id"][local_end - 1])
    current_chunk_start = int(index["chunk_start"][c])
    player_first_chunk = int(index["player_first_chunk"][local_end - 1])

    chunk_ranges = []
    if local_end > current_chunk_start:
        chunk_ranges.append((current_chunk_start, local_end))
    next_c = c - 1
    while next_c >= player_first_chunk and len(chunk_ranges) < max_chunks:
        chunk_ranges.append((int(index["chunk_start"][next_c]), int(index["chunk_end"][next_c])))
        next_c -= 1
    chunk_ranges.reverse()

    resolved = []
    for s, e in chunk_ranges:
        length = min(e - s, max_pitch_len)
        local_start = e - length
        days_before_cutoff = float((cutoff_date_ns - index["game_date_ns"][e - 1]) / NS_PER_DAY)
        resolved.append((local_start, e, days_before_cutoff))
    return resolved


class QueryChunkedHistoryDataset(Dataset):
    """One sample per (player_id, cutoff_date_ns) query pair -- unlike
    NextPitchLongHistoryDataset, `queries` doesn't have to be existing pitch
    rows, so the same (player, date) pair used by many pitches only needs
    one entry here."""

    def __init__(self, index: dict, queries: list[tuple[int, int]], max_chunks: int, max_pitch_len: int) -> None:
        self.index = index
        self.queries = queries
        self.max_chunks = max_chunks
        self.max_pitch_len = max_pitch_len

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict:
        player_id_value, cutoff_ns = self.queries[idx]
        resolved = chunk_ranges_for_query(self.index, player_id_value, cutoff_ns, self.max_chunks, self.max_pitch_len)
        if not resolved:
            return {"has_history": False, "num_chunks": 0, "chunks": [], "target": 0}
        chunks = []
        for local_start, end, days_before_cutoff in resolved:
            chunks.append(
                {
                    "length": end - local_start,
                    "continuous": torch.from_numpy(self.index["continuous"][local_start:end]),
                    "pitch_type": torch.from_numpy(self.index["pitch_type_idx"][local_start:end]),
                    "outcome": torch.from_numpy(self.index["outcome_idx"][local_start:end]),
                    "matchup": torch.from_numpy(self.index["matchup_idx"][local_start:end]),
                    "days_before_cutoff": days_before_cutoff,
                }
            )
        return {"has_history": True, "num_chunks": len(chunks), "chunks": chunks, "target": 0}


# ---------------------------------------------------------------------------
# Precompute + on-disk cache.
# ---------------------------------------------------------------------------


def distinct_pairs(pitches: pd.DataFrame, perspective: str) -> list[tuple[int, int]]:
    """Every distinct (player_id, game_date) pair for this perspective,
    across the whole table -- one embedding lookup per pair, not per pitch."""
    id_column = f"{perspective}_id"
    unique = pitches.drop_duplicates([id_column, "game_date"])[[id_column, "game_date"]]
    return [(int(r[0]), pd.Timestamp(r[1]).value) for r in unique.itertuples(index=False)]


def roster_active_pairs(
    pitcher_appearances: pd.DataFrame, team_dates: pd.DataFrame, window_days: int
) -> list[tuple[int, int]]:
    """Every (pitcher_id, game_date_ns) pair where that pitcher appeared for
    the *same team* within `window_days` strictly before `game_date` --
    broader than distinct_pairs' "dates they personally threw a pitch that
    day," and the actual gap distinct_pairs leaves uncovered: a player's
    long-history embedding depends only on their own history strictly
    before the query date (see LongHistoryEncoder / chunk_ranges_for_query,
    which never require the cutoff itself to coincide with one of the
    player's own pitch dates -- nothing here computes anything *at* the
    cutoff, only what precedes it), so there's no technical reason an
    embedding can only be queried for a date the player personally pitched.

    "Roster-active" has no real 26-man-roster data source behind it
    anywhere in this project (see bullpen_availability.py's own docstring
    on the same limitation) -- this reuses that exact same trailing-
    appearance-window proxy (a pitcher who appeared for a team recently is
    treated as still roster-active for that team), just as a *cache-
    building* query generator rather than a runtime availability score.

    `pitcher_appearances`: game_pk/team/pitcher_id/game_date (see
    src/data/game_dataset.py). `team_dates`: the (team, game_date) pairs to
    generate candidate embeddings for -- typically every date some game
    actually happened for that team, so the pairs this returns line up with
    real games a caller might want to simulate a hypothetical reliever
    appearance in.
    """
    pairs: list[tuple[int, int]] = []
    window_ns = window_days * NS_PER_DAY
    for team, group in pitcher_appearances.sort_values("game_date").groupby("team"):
        dates_ns = group["game_date"].to_numpy().astype("datetime64[ns]").astype("int64")
        pitcher_ids = group["pitcher_id"].to_numpy()

        team_target_dates = team_dates.loc[team_dates["team"] == team, "game_date"]
        for target_date in team_target_dates:
            cutoff_ns = pd.Timestamp(target_date).value
            window_mask = (dates_ns >= cutoff_ns - window_ns) & (dates_ns < cutoff_ns)
            for pid in set(pitcher_ids[window_mask].tolist()):
                pairs.append((int(pid), cutoff_ns))
    return pairs


def _old_format_player_path(perspective_dir: Path, player_id_value: int) -> Path:
    """Legacy one-dict-per-player cache file. Still read (for the
    already-cached check and by EmbeddingCache as a fallback) but never
    written by current code -- see module docstring."""
    return perspective_dir / f"{player_id_value}.pt"


def _entry_path(perspective_dir: Path, player_id_value: int, date_ns: int) -> Path:
    """Current one-file-per-(player, date) cache entry."""
    return perspective_dir / str(player_id_value) / f"{date_ns}.pt"


def _already_cached_dates(perspective_dir: Path, player_id_value: int) -> set[int]:
    """Every date_ns already cached for this player, old-format dict keys
    unioned with new-format entry filenames -- without loading any
    embedding tensors, just enough to know what's missing."""
    dates: set[int] = set()
    old_path = _old_format_player_path(perspective_dir, player_id_value)
    if old_path.exists():
        dates.update(torch.load(old_path, weights_only=False).keys())
    entry_dir = perspective_dir / str(player_id_value)
    if entry_dir.exists():
        dates.update(int(p.stem) for p in entry_dir.glob("*.pt"))
    return dates


def precompute_and_cache_embeddings(
    pitches: pd.DataFrame,
    encoder: LongHistoryEncoder,
    cache_dir: Path,
    max_chunks: int,
    max_pitch_len: int,
    device: torch.device,
    batch_size: int = 32,
    perspectives: tuple[str, ...] = PERSPECTIVES,
    queries_by_perspective: dict[str, list[tuple[int, int]]] | None = None,
) -> dict[str, int]:
    """Computes every requested (player_id, game_date) pair's embedding not
    already cached, for each perspective, and writes them to disk -- one
    small file per entry under cache_dir/{perspective}/{player_id}/{date_ns}.pt
    (old-format per-player dict files are read but never rewritten -- see
    module docstring). An interrupted or re-run call only computes what's
    missing, same resume convention as PlayerPitchSequenceDataset.precompute_and_cache.
    Returns {perspective: number of pairs actually computed}.

    `queries_by_perspective`, if given, overrides distinct_pairs' default
    query list for whichever perspectives it names (any perspective not
    named there still falls back to distinct_pairs -- "every date this
    player personally pitched/batted"). Use this to extend coverage beyond
    that default, e.g. roster_active_pairs' broader "any date this pitcher
    was recently active for this team" query set -- `pitches` still supplies
    each player's real chunked history (build_chunk_index below), so a
    query date doesn't need to be one of that player's own real pitch dates
    for its embedding to be computed correctly (see roster_active_pairs'
    docstring for why that's true of the underlying encoder machinery).

    Writes are incremental, one flush per entry as soon as that entry's
    embedding is available, rather than deferred to the end of a batch or
    perspective's loop. A batch near the largest chunk-count/pitch-length
    combination can take a long time (or, if it hangs -- see git history
    around this module -- may never finish); this way a kill/crash/OOM only
    loses whatever entries in the current batch hadn't been written yet, not
    everything computed since the perspective started.
    """
    encoder = encoder.to(device)
    encoder.eval()
    counts: dict[str, int] = {}
    queries_by_perspective = queries_by_perspective or {}

    for perspective in perspectives:
        id_column = f"{perspective}_id"
        perspective_dir = Path(cache_dir) / perspective
        perspective_dir.mkdir(parents=True, exist_ok=True)

        queries = queries_by_perspective.get(perspective) or distinct_pairs(pitches, perspective)
        logger.info("%s: %d distinct (player, date) pairs total", perspective, len(queries))

        # Per player, only load *which dates* are already cached (old-format
        # dict keys unioned with new-format entry filenames) -- not the
        # embedding tensors themselves, since nothing here rewrites the
        # old-format files and the new format never needs a full player dict
        # in memory at all.
        players_needed = sorted({q[0] for q in queries})
        already_cached = {p: _already_cached_dates(perspective_dir, p) for p in players_needed}

        remaining = [
            (player_id_value, date_ns)
            for player_id_value, date_ns in queries
            if date_ns not in already_cached[player_id_value]
        ]
        logger.info("%s: %d pairs already cached, %d remaining to compute", perspective, len(queries) - len(remaining), len(remaining))

        if remaining:
            index = build_chunk_index(pitches, id_column)
            dataset = QueryChunkedHistoryDataset(index, remaining, max_chunks, max_pitch_len)
            num_chunks_per_sample = estimate_num_chunks(index, remaining, max_chunks)
            sampler = BucketByChunkCountSampler(num_chunks_per_sample, batch_size, shuffle=False)
            loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_long_history_batch)

            t0 = time.time()
            n_done = 0
            touched_dirs: set[Path] = set()
            with torch.no_grad():
                for batch_indices, (chunk_pitch_sequences, days_before_cutoff, chunk_padding_mask, has_history, _) in zip(
                    sampler, loader
                ):
                    chunk_pitch_sequences = {k: v.to(device) for k, v in chunk_pitch_sequences.items()}
                    days_before_cutoff = days_before_cutoff.to(device)
                    chunk_padding_mask = chunk_padding_mask.to(device)
                    has_history = has_history.to(device)
                    out = encoder(chunk_pitch_sequences, days_before_cutoff, chunk_padding_mask, has_history).cpu()

                    # Flush immediately, one small file per entry -- unlike
                    # the old per-player dict, each write's cost is
                    # independent of that player's accumulated history size.
                    for local_i, global_i in enumerate(batch_indices):
                        player_id_value, date_ns = remaining[global_i]
                        path = _entry_path(perspective_dir, player_id_value, date_ns)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(out[local_i], path)
                        touched_dirs.add(path.parent)

                    n_done += len(batch_indices)
                    if n_done % (batch_size * 200) == 0:
                        elapsed = time.time() - t0
                        logger.info(
                            "%s: %d/%d computed (%.1f pairs/s)", perspective, n_done, len(remaining), n_done / max(elapsed, 1e-9)
                        )

            logger.info("%s: wrote entries for %d player(s)", perspective, len(touched_dirs))

        counts[perspective] = len(remaining)

    return counts


class EmbeddingCache:
    """Read-only lookup over an on-disk cache built by
    precompute_and_cache_embeddings. Loads a player's old-format dict file
    (if any) lazily on first lookup and keeps it in memory -- same
    lazy-load-then-memoize convention as
    PlayerPitchSequenceDataset._ensure_player_cache_loaded -- then falls
    back to the new one-file-per-entry format on a miss, memoizing whatever
    it loads from there too so a repeated lookup never re-hits disk.

    Never computes an embedding itself: a miss in both formats raises
    KeyError rather than silently falling back to a live encoder forward
    pass, since that fallback would defeat the entire point of precomputing
    (see module docstring) and would mask real cache-build gaps instead of
    surfacing them.
    """

    def __init__(self, cache_dir: Path, perspective: str) -> None:
        if perspective not in PERSPECTIVES:
            raise ValueError(f"perspective must be one of {PERSPECTIVES}, got {perspective!r}")
        self.perspective = perspective
        self.cache_dir = Path(cache_dir) / perspective
        self._loaded_players: set = set()
        self._memory_cache: dict[int, dict[int, torch.Tensor]] = {}

    def _ensure_loaded(self, player_id_value: int) -> None:
        if player_id_value in self._loaded_players:
            return
        self._loaded_players.add(player_id_value)
        path = _old_format_player_path(self.cache_dir, player_id_value)
        if path.exists():
            self._memory_cache[player_id_value] = torch.load(path, weights_only=False)
        else:
            self._memory_cache[player_id_value] = {}

    def get(self, player_id_value, game_date) -> torch.Tensor:
        player_id_value = int(player_id_value)
        cutoff = pd.Timestamp(game_date)
        self._ensure_loaded(player_id_value)
        cache = self._memory_cache[player_id_value]
        embedding = cache.get(cutoff.value)
        if embedding is None:
            entry_path = _entry_path(self.cache_dir, player_id_value, cutoff.value)
            if entry_path.exists():
                embedding = torch.load(entry_path, weights_only=False)
                cache[cutoff.value] = embedding
        if embedding is None:
            raise KeyError(
                f"No cached {self.perspective} embedding for player_id={player_id_value}, "
                f"game_date={cutoff.date()} in {self.cache_dir}. Run precompute_and_cache_embeddings "
                "for this (player, date) pair first -- this cache never computes on a miss."
            )
        return embedding

    def get_batch(self, player_ids: pd.Series, game_dates: pd.Series) -> torch.Tensor:
        """Vectorized-in-name-only convenience: still one get() per row
        (each is an O(1) dict lookup once that player's file is loaded), but
        saves the caller writing the same zip/stack loop everywhere."""
        return torch.stack([self.get(p, d) for p, d in zip(player_ids, game_dates)])


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute and cache LongHistoryEncoder embeddings for the event model.")
    parser.add_argument("--pitches-dir", type=Path, default=PROCESSED_DATA_DIR / "pitches")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument(
        "--season-start", type=int, default=min(TRAIN_SEASON_RANGE[0], VAL_SEASONS[0], TEST_SEASON_RANGE[0])
    )
    # No hardcoded default end season: TEST_SEASON_RANGE's end (2025) is a
    # deliberate fixed modeling holdout, not "the latest data we have" --
    # pinning to it here would silently drop whatever partial current season
    # (e.g. 2026) has already been pulled every time this cache is (re)built.
    # None means "through the latest season actually present in the data,"
    # resolved once that's loaded (see main()).
    parser.add_argument("--season-end", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)

    full = read_partitioned(args.pitches_dir)
    season_end = args.season_end if args.season_end is not None else int(full["season"].max())
    logger.info("Loading pitches from %s (seasons %d-%d)", args.pitches_dir, args.season_start, season_end)
    pitches = full[full["season"].between(args.season_start, season_end) & full["is_valid"]].reset_index(drop=True)
    logger.info("%d pitches", len(pitches))

    logger.info("Loading frozen LongHistoryEncoder checkpoint from %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    chunk_config = ChunkEncoderConfig(**ckpt["chunk_config"])
    career_config = CareerEncoderConfig(**ckpt["career_config"])
    chunk_encoder = ChunkEncoder(chunk_config)
    career_encoder = CareerEncoder(career_config)
    chunk_encoder.load_state_dict(ckpt["chunk_encoder_state_dict"])
    career_encoder.load_state_dict(ckpt["career_encoder_state_dict"])
    encoder = LongHistoryEncoder(chunk_encoder, career_encoder)

    t0 = time.time()
    counts = precompute_and_cache_embeddings(
        pitches, encoder, args.cache_dir, career_config.max_chunks, chunk_config.max_seq_len, device, args.batch_size
    )
    logger.info("Done in %.1fs. Computed: %s. Cache at %s", time.time() - t0, counts, args.cache_dir)


if __name__ == "__main__":
    main()
