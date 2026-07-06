"""Per-player pitch sequence dataset, plus a non-sequence fallback for cold starts.

Built on top of the cleaned pitch table produced by build_features.py (see
statcast_common.read_partitioned for how to load data/processed/pitches/). A
"player" is either the pitcher who threw the pitch or the batter who saw it,
selected via `perspective` -- both share the same schema so the same class
works for either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.statcast_common import DESCRIPTION_OUTCOME_MAP, EVENT_OUTCOME_MAP

CONTINUOUS_FEATURES = ["release_speed", "spin_rate", "plate_x", "plate_z"]

# Grounded in the pitch_type codes actually present in data/processed/pitches.
# "UNK" covers nulls and any future code not in this list, rather than crashing.
PITCH_TYPE_VOCAB = [
    "AB", "CH", "CS", "CU", "EP", "FA", "FC", "FF", "FO", "FS",
    "IN", "KC", "KN", "PO", "SC", "SI", "SL", "ST", "SV", "UN", "UNK",
]
# Derived from statcast_common's outcome maps so this can't drift out of sync
# with the labels compute_outcome() actually produces. "UNK" covers the rare
# pitch flagged invalid for a missing/unmapped outcome (see flag_missing_critical).
OUTCOME_VOCAB = sorted(set(EVENT_OUTCOME_MAP.values()) | set(DESCRIPTION_OUTCOME_MAP.values())) + ["UNK"]
# Batter-stand / pitcher-throws matchup. Statcast only ever records R/L for
# either side, so this is an exhaustive 2x2 plus an UNK fallback.
MATCHUP_VOCAB = ["R_R", "R_L", "L_R", "L_L", "UNK"]

PITCH_TYPE_INDEX = {v: i for i, v in enumerate(PITCH_TYPE_VOCAB)}
OUTCOME_INDEX = {v: i for i, v in enumerate(OUTCOME_VOCAB)}
MATCHUP_INDEX = {v: i for i, v in enumerate(MATCHUP_VOCAB)}


def category_indices(values: pd.Series, index: dict[str, int]) -> torch.Tensor:
    """Map a Series of category labels to integer indices via `index`, sending
    nulls and anything unrecognized to `index["UNK"]`. Shared with
    pretrain_encoder.py's NextPitchDataset so both use identical vocab handling."""
    unk = index["UNK"]
    return torch.tensor([index.get(v, unk) if pd.notna(v) else unk for v in values], dtype=torch.long)


def _empty_sequence(player_id, cutoff_date: pd.Timestamp) -> dict:
    return {
        "player_id": player_id,
        "cutoff_date": cutoff_date,
        "has_history": False,
        "length": 0,
        "continuous": torch.zeros((0, len(CONTINUOUS_FEATURES)), dtype=torch.float32),
        "pitch_type": torch.zeros((0,), dtype=torch.long),
        "outcome": torch.zeros((0,), dtype=torch.long),
        "matchup": torch.zeros((0,), dtype=torch.long),
        "position": torch.zeros((0,), dtype=torch.long),
    }


class PlayerPitchSequenceDataset(Dataset):
    """One sample = one (player_id, cutoff_date) pair.

    Returns that player's pitches strictly before the cutoff date, most recent
    last, truncated to the most recent `max_seq_len` if there's more history
    than that. Sequences are returned at their actual length (<= max_seq_len)
    and are NOT padded -- pad in a collate_fn if batching more than one sample.

    Pitches are sorted once at construction time by (id_column, game_date,
    at_bat_number, pitch_number) and indexed into per-player row ranges (see
    _build_player_ranges), so build_sequence is an O(log k) binary search
    over that one player's own rows (k = that player's total pitch count)
    plus an O(max_seq_len) slice, rather than an O(n) scan of the whole
    table on every call. That matters a lot here: profiling
    GameOutcomeDataset showed the old per-call `pitches[id_col == player_id]`
    full-table boolean mask -- doubly slow since pitcher_id/batter_id are
    pandas nullable Int64 columns, so pandas runs the null-aware "kleene"
    comparison/AND path instead of plain numpy boolean ops -- was the
    dominant cost of building a single game's training sample (~1.1s/game,
    the vast majority of it here rather than in tensor construction).

    If `cache_dir` is given, build_sequence also checks an on-disk cache
    (one file per player under that directory) before computing from
    scratch -- but it never *writes* to that cache itself, since doing so
    from multiple concurrent DataLoader worker processes would race. Call
    `precompute_and_cache` once, single-process, before training to
    populate it (see GameOutcomeDataset.warm_cache), then every epoch (and
    every later process, e.g. a backtest run against the same cache_dir)
    just reads it.
    """

    def __init__(
        self,
        pitches: pd.DataFrame,
        samples: list[tuple[int, "str | pd.Timestamp"]],
        max_seq_len: int,
        perspective: Literal["pitcher", "batter"] = "pitcher",
        continuous_stats: dict[str, tuple[float, float]] | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        if perspective not in ("pitcher", "batter"):
            raise ValueError(f"perspective must be 'pitcher' or 'batter', got {perspective!r}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        self.perspective = perspective
        self.id_column = f"{perspective}_id"
        self.max_seq_len = max_seq_len
        self.samples = samples
        self.cache_dir = Path(cache_dir) / perspective if cache_dir is not None else None
        self.pitches = pitches.sort_values(
            [self.id_column, "game_date", "at_bat_number", "pitch_number"]
        ).reset_index(drop=True)
        self.continuous_stats = continuous_stats or self._compute_continuous_stats(pitches)

        self._player_ranges = self._build_player_ranges()
        # int64 nanoseconds, matching pd.Timestamp.value's unit exactly --
        # game_date's own on-disk unit varies (seen both us and ns), so this
        # normalizes both sides of every binary search to the same unit.
        self._dates_ns = self.pitches["game_date"].to_numpy().astype("datetime64[ns]").astype("int64")

        self._loaded_players: set = set()
        self._memory_cache: dict = {}

    def _build_player_ranges(self) -> dict:
        """Maps player_id -> (start, end) row range in self.pitches. Cheap:
        self.pitches is already sorted by id_column first, so each player's
        rows are one contiguous block; this just finds the block boundaries."""
        ids = self.pitches[self.id_column].to_numpy()
        if len(ids) == 0:
            return {}
        boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(ids)]))
        return dict(zip(ids[starts].tolist(), zip(starts.tolist(), ends.tolist())))

    @staticmethod
    def _compute_continuous_stats(pitches: pd.DataFrame) -> dict[str, tuple[float, float]]:
        stats = {}
        for col in CONTINUOUS_FEATURES:
            values = pitches[col].to_numpy(dtype="float64", na_value=np.nan)
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values))
            stats[col] = (mean, std if std > 0 else 1.0)
        return stats

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        player_id, cutoff_date = self.samples[idx]
        return self.build_sequence(player_id, cutoff_date)

    def build_sequence(self, player_id, cutoff_date) -> dict:
        cutoff_date = pd.Timestamp(cutoff_date)

        if self.cache_dir is not None:
            self._ensure_player_cache_loaded(player_id)
            cached = self._memory_cache.get(player_id, {}).get(cutoff_date.value)
            if cached is not None:
                return cached

        return self._compute_sequence(player_id, cutoff_date)

    def _compute_sequence(self, player_id, cutoff_date: pd.Timestamp) -> dict:
        range_ = self._player_ranges.get(player_id)
        if range_ is None:
            return _empty_sequence(player_id, cutoff_date)

        start, end = range_
        # Number of this player's rows strictly before cutoff_date: dates_ns
        # is ascending within [start, end), so this is exactly the "how many
        # of this player's pitches precede the cutoff" count.
        preceding_count = int(np.searchsorted(self._dates_ns[start:end], cutoff_date.value, side="left"))
        length = min(preceding_count, self.max_seq_len)
        if length <= 0:
            return _empty_sequence(player_id, cutoff_date)

        local_end = start + preceding_count
        local_start = local_end - length
        history = self.pitches.iloc[local_start:local_end]

        continuous = np.stack(
            [
                (history[col].to_numpy(dtype="float64", na_value=np.nan) - mean) / std
                for col, (mean, std) in self.continuous_stats.items()
            ],
            axis=1,
        )
        continuous = np.nan_to_num(continuous, nan=0.0)

        matchup = history["stand"].astype(object) + "_" + history["p_throws"].astype(object)

        return {
            "player_id": player_id,
            "cutoff_date": cutoff_date,
            "has_history": True,
            "length": length,
            "continuous": torch.tensor(continuous, dtype=torch.float32),
            "pitch_type": category_indices(history["pitch_type"], PITCH_TYPE_INDEX),
            "outcome": category_indices(history["outcome"], OUTCOME_INDEX),
            "matchup": category_indices(matchup, MATCHUP_INDEX),
            "position": torch.arange(length, dtype=torch.long),
        }

    def _cache_path(self, player_id) -> Path:
        return self.cache_dir / f"{player_id}.pt"

    def _ensure_player_cache_loaded(self, player_id) -> None:
        if player_id in self._loaded_players:
            return
        self._loaded_players.add(player_id)
        path = self._cache_path(player_id)
        if path.exists():
            self._memory_cache[player_id] = torch.load(path, weights_only=False)
        else:
            self._memory_cache.setdefault(player_id, {})

    def precompute_and_cache(self, queries: list[tuple]) -> int:
        """Computes every (player_id, cutoff_date) pair in `queries` not
        already cached and writes them to disk, one file per player (merged
        with whatever that player's file already held, so repeated warm
        calls accumulate rather than clobber). Meant to be called once,
        single-process, before training/inference starts (see
        GameOutcomeDataset.warm_cache) -- build_sequence itself never
        writes, which is what makes it safe to read this same cache_dir
        from multiple DataLoader worker processes afterwards. Returns the
        number of sequences actually computed (as opposed to already-cached).
        """
        if self.cache_dir is None:
            raise ValueError("cache_dir was not set on this dataset")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        dirty_players = set()
        computed = 0
        for player_id, cutoff_date in queries:
            cutoff_date = pd.Timestamp(cutoff_date)
            self._ensure_player_cache_loaded(player_id)
            player_cache = self._memory_cache.setdefault(player_id, {})
            if cutoff_date.value in player_cache:
                continue
            player_cache[cutoff_date.value] = self._compute_sequence(player_id, cutoff_date)
            dirty_players.add(player_id)
            computed += 1

        for player_id in dirty_players:
            torch.save(self._memory_cache[player_id], self._cache_path(player_id))

        return computed


class FallbackPlayerFeatures:
    """Non-sequence features for cold-start players (zero pitch history before
    the cutoff date, i.e. has_history=False from PlayerPitchSequenceDataset).

    Age is computed from `player_bio` if one is supplied; there's no
    biographical or minor-league data source wired up yet, so `age` comes back
    NaN and `minor_league_stats` stays a placeholder until one exists.
    """

    def __init__(self, player_bio: pd.DataFrame | None = None) -> None:
        # player_bio expected columns: player_id, birth_date
        self.player_bio = player_bio

    def get_features(self, player_id, cutoff_date) -> dict:
        cutoff_date = pd.Timestamp(cutoff_date)
        return {
            "player_id": player_id,
            "cutoff_date": cutoff_date,
            "age": self._compute_age(player_id, cutoff_date),
            # Reserved for future features (e.g. minor-league performance
            # stats) once a data source for them exists.
            "minor_league_stats": None,
        }

    def _compute_age(self, player_id, cutoff_date: pd.Timestamp) -> float:
        if self.player_bio is None:
            return float("nan")
        match = self.player_bio.loc[self.player_bio["player_id"] == player_id, "birth_date"]
        if match.empty or pd.isna(match.iloc[0]):
            return float("nan")
        birth_date = pd.Timestamp(match.iloc[0])
        return (cutoff_date - birth_date).days / 365.25
