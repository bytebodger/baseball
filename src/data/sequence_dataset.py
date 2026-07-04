"""Per-player pitch sequence dataset, plus a non-sequence fallback for cold starts.

Built on top of the cleaned pitch table produced by build_features.py (see
statcast_common.read_partitioned for how to load data/processed/pitches/). A
"player" is either the pitcher who threw the pitch or the batter who saw it,
selected via `perspective` -- both share the same schema so the same class
works for either.
"""

from __future__ import annotations

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
    """

    def __init__(
        self,
        pitches: pd.DataFrame,
        samples: list[tuple[int, "str | pd.Timestamp"]],
        max_seq_len: int,
        perspective: Literal["pitcher", "batter"] = "pitcher",
        continuous_stats: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if perspective not in ("pitcher", "batter"):
            raise ValueError(f"perspective must be 'pitcher' or 'batter', got {perspective!r}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        self.id_column = f"{perspective}_id"
        self.max_seq_len = max_seq_len
        self.samples = samples
        self.pitches = pitches.sort_values(["game_date", "at_bat_number", "pitch_number"]).reset_index(drop=True)
        self.continuous_stats = continuous_stats or self._compute_continuous_stats(pitches)

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
        history = self.pitches[
            (self.pitches[self.id_column] == player_id) & (self.pitches["game_date"] < cutoff_date)
        ]

        if history.empty:
            return _empty_sequence(player_id, cutoff_date)

        # Already sorted chronologically ascending; keep only the most recent
        # max_seq_len rows so the most recent pitch ends up last.
        history = history.tail(self.max_seq_len)
        length = len(history)

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
