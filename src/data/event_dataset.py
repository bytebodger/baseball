"""Per-pitch dataset for the event model (Phase 4): predicts a single
pitch's outcome (see OUTCOME_VOCAB, src/data/sequence_dataset.py) from

- the pitcher's and batter's precomputed long-history embeddings as of that
  pitch's own game_date (src/data/event_embedding_cache.py) -- "as of"
  meaning strictly before, the same leak-safe convention the embedding
  cache itself enforces (see chunk_ranges_for_query);
- situational context at the moment of the pitch (count, outs, base state,
  score differential, inning, times through the order, handedness matchup);
- that game's park factor embedding and the league-wide rolling rates
  (src/data/park_factors.py), both keyed by (park_id, season)/season and
  themselves already leak-safe (strictly-prior-seasons rolling windows);
- the pitcher's and batter's rolling, strictly-prior batted-ball-quality
  history (src/data/contact_quality.py): average exit velocity and hard-hit
  rate allowed/produced, built from raw Statcast launch_speed -- a direct,
  explicit signal for contact quality that the long-history encoder's own
  embedding has no equivalent of (it only ever sees the coarse
  hit_into_play_out/single/double/... outcome category per pitch, never a
  batted ball's actual exit velocity).

One situational feature the original Phase 4 spec called for isn't
available from the processed pitch table (data/processed/pitches, built by
build_features.py) and is left out here rather than faked: day/night -- no
start-time or day/night flag exists anywhere in this pipeline, raw or
processed.

score_diff is batting-team-relative (positive when the team currently at
bat is ahead), via inning_topbot (Top = away team batting, Bot = home team
batting) -- joined onto data/processed/pitches from the raw Statcast files
on (game_pk, at_bat_number, pitch_number), a clean, purely additive 1:1 join
(see git history around this module for the join script and its integrity
checks) rather than a full rebuild of the processed pitch table. An earlier
version of this module used home_score - away_score instead (no half-inning
information was available yet) -- still informative as a blowout-vs-close
signal, but not perspective-relative the way an actual batting team's run
differential is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.contact_quality import (
    MIN_BATTED_BALLS_FOR_STABLE_ESTIMATE,
    ContactQualityHistory,
    babip_features_batch,
    contact_quality_features_batch,
)
from src.data.event_embedding_cache import EmbeddingCache
from src.data.park_factors import ParkFactorEmbedding, league_rates_for
from src.data.sequence_dataset import MATCHUP_INDEX, OUTCOME_INDEX, category_indices

# balls/strikes/outs_when_up are guaranteed non-null on is_valid rows (all
# three are critical fields in build_features.py); score_diff is derived,
# never null as long as home_score/away_score (also critical) are present.
SITUATIONAL_CONTINUOUS_FEATURES = ["balls", "strikes", "outs_when_up", "score_diff", "inning", "times_through_order"]
# Deliberately NOT critical fields (see build_features.py): null here means
# "no runner on that base," a legitimate value, not missing data.
BASE_STATE_COLUMNS = ["on_1b", "on_2b", "on_3b"]
# pitcher exit-velo-allowed (z-scored) + pitcher hard-hit-rate-allowed +
# batter exit-velo-produced (z-scored) + batter hard-hit-rate-produced --
# see contact_quality.py. Hard-hit rate is already a 0-1 rate (same scale as
# the base-state flags below), left un-scaled the same way league_rates is;
# exit velocity is z-scored (contact_quality_stats) since its raw ~80-95mph
# scale would otherwise dominate the other, already-normalized context
# scalars feeding the same linear layer.
#
# Folded directly into the same "context" tensor as the situational/base-
# state/league-rate scalars (see EventBatchCollator below), not routed
# through a separate sub-network -- an earlier version gave this its own
# dedicated 2-layer MLP before concatenation, but that measurably regressed
# the 35-game low-scoring-game calibration check versus this simpler
# raw-scalar version (see event_model.py's module docstring), so it was
# reverted.
CONTACT_QUALITY_FEATURE_NAMES = ["pitcher_exit_velo", "pitcher_hard_hit_rate", "batter_exit_velo", "batter_hard_hit_rate"]
# The scalar width of EventBatchCollator's "context" tensor -- the 6
# z-scored situational features + 3 base-occupied flags + 2 league rolling
# rates + 4 contact-quality features. EventModelConfig.situational_dim must
# match this.
CONTEXT_DIM = len(SITUATIONAL_CONTINUOUS_FEATURES) + len(BASE_STATE_COLUMNS) + 2 + len(CONTACT_QUALITY_FEATURE_NAMES)
# Real, leak-safe (pitcher_babip_allowed, pitcher_hard_hit_rate_allowed)
# targets for EventModel.contact_quality_aux_head's auxiliary regression
# loss (see event_model.py's module docstring and
# contact_quality.contact_quality_aux_targets_batch) -- raw values, not
# z-scored, since both are already naturally 0-1-scaled and MSE against
# them is numerically stable without it.
CONTACT_QUALITY_AUX_TARGET_NAMES = ["pitcher_babip", "pitcher_hard_hit_rate"]


def _with_score_diff(pitches: pd.DataFrame) -> pd.DataFrame:
    """Batting-team-relative score differential: positive when the team
    currently at bat is ahead. inning_topbot == "Top" means the away team is
    batting (the home team is fielding) and vice versa for "Bot"."""
    is_away_batting = pitches["inning_topbot"] == "Top"
    batting_score = pitches["away_score"].where(is_away_batting, pitches["home_score"])
    fielding_score = pitches["home_score"].where(is_away_batting, pitches["away_score"])
    return pitches.assign(score_diff=batting_score - fielding_score)


def compute_situational_stats(pitches: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Train-only mean/std for SITUATIONAL_CONTINUOUS_FEATURES -- same
    z-score-from-train-split convention as
    PlayerPitchSequenceDataset._compute_continuous_stats, so situational
    features are normalized without ever looking at val/test rows."""
    frame = _with_score_diff(pitches)
    stats = {}
    for col in SITUATIONAL_CONTINUOUS_FEATURES:
        values = frame[col].to_numpy(dtype="float64", na_value=np.nan)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        stats[col] = (mean, std if std > 0 else 1.0)
    return stats


def compute_contact_quality_stats(
    pitches: pd.DataFrame,
    pitcher_contact_quality: ContactQualityHistory,
    batter_contact_quality: ContactQualityHistory,
    min_events: int = MIN_BATTED_BALLS_FOR_STABLE_ESTIMATE,
) -> dict[str, tuple[float, float]]:
    """Train-only mean/std for the two exit-velocity contact-quality
    features (pitcher-allowed, batter-produced) -- same z-score-from-train-
    split convention as compute_situational_stats, computed over the
    per-row rolling values `pitches` itself would get (already leak-safe by
    construction: each row's own value only ever reflects that player's
    history strictly before that row's own game_date), not over the raw
    batted-ball population directly. `min_events` must match whatever
    EventDataset itself is built with, or the z-scoring stats and the
    features they normalize won't agree on which rows are real vs.
    league-average fallbacks."""
    pitcher_features = contact_quality_features_batch(pitcher_contact_quality, pitches["pitcher_id"], pitches["game_date"], min_events)
    batter_features = contact_quality_features_batch(batter_contact_quality, pitches["batter_id"], pitches["game_date"], min_events)

    def _stats(values: np.ndarray) -> tuple[float, float]:
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        return (mean, std if std > 0 else 1.0)

    return {"pitcher_exit_velo": _stats(pitcher_features[:, 0]), "batter_exit_velo": _stats(batter_features[:, 0])}


class EventDataset(Dataset):
    """One sample per pitch. Every cheap per-row array (situational
    features, base-state flags, matchup index, park factor index, league
    rates, target outcome index) is precomputed once in __init__ -- the only
    work left per batch is the embedding cache lookup, which is batched in
    EventBatchCollator instead of repeated len(dataset) times through a slow
    per-item path.
    """

    def __init__(
        self,
        pitches: pd.DataFrame,
        situational_stats: dict[str, tuple[float, float]],
        park_factor_embedding: ParkFactorEmbedding,
        league_rates: pd.DataFrame,
        pitcher_contact_quality: ContactQualityHistory,
        batter_contact_quality: ContactQualityHistory,
        contact_quality_stats: dict[str, tuple[float, float]],
        contact_quality_min_events: int = MIN_BATTED_BALLS_FOR_STABLE_ESTIMATE,
    ) -> None:
        self.pitches = pitches.reset_index(drop=True)
        self.pitcher_ids = self.pitches["pitcher_id"].to_numpy()
        self.batter_ids = self.pitches["batter_id"].to_numpy()
        self.game_dates = self.pitches["game_date"]

        frame = _with_score_diff(self.pitches)
        continuous = np.stack(
            [
                (frame[col].to_numpy(dtype="float64", na_value=np.nan) - mean) / std
                for col, (mean, std) in situational_stats.items()
            ],
            axis=1,
        )
        self.situational = torch.tensor(np.nan_to_num(continuous, nan=0.0), dtype=torch.float32)

        base_state = np.stack([self.pitches[c].notna().to_numpy() for c in BASE_STATE_COLUMNS], axis=1)
        self.base_state = torch.tensor(base_state, dtype=torch.float32)

        matchup = self.pitches["stand"].astype(object) + "_" + self.pitches["p_throws"].astype(object)
        self.matchup_index = category_indices(matchup, MATCHUP_INDEX)

        self.park_index = park_factor_embedding.indices_for(self.pitches["park_id"], self.pitches["season"])
        rates = league_rates_for(self.pitches["season"], league_rates)
        self.league_rates = torch.tensor(rates.to_numpy(dtype="float32"))

        pitcher_contact = contact_quality_features_batch(
            pitcher_contact_quality, self.pitches["pitcher_id"], self.pitches["game_date"], contact_quality_min_events
        )
        batter_contact = contact_quality_features_batch(
            batter_contact_quality, self.pitches["batter_id"], self.pitches["game_date"], contact_quality_min_events
        )
        pitcher_mean, pitcher_std = contact_quality_stats["pitcher_exit_velo"]
        batter_mean, batter_std = contact_quality_stats["batter_exit_velo"]
        contact_quality = np.stack(
            [
                (pitcher_contact[:, 0] - pitcher_mean) / pitcher_std,
                pitcher_contact[:, 1],
                (batter_contact[:, 0] - batter_mean) / batter_std,
                batter_contact[:, 1],
            ],
            axis=1,
        )
        self.contact_quality = torch.tensor(contact_quality, dtype=torch.float32)

        pitcher_babip = babip_features_batch(
            pitcher_contact_quality, self.pitches["pitcher_id"], self.pitches["game_date"], contact_quality_min_events
        )
        aux_targets = np.stack([pitcher_babip, pitcher_contact[:, 1]], axis=1)
        self.contact_quality_aux_target = torch.tensor(aux_targets, dtype=torch.float32)

        self.target = category_indices(self.pitches["outcome"], OUTCOME_INDEX)

    def __len__(self) -> int:
        return len(self.pitches)

    def __getitem__(self, idx: int) -> dict:
        return {
            "pitcher_id": int(self.pitcher_ids[idx]),
            "batter_id": int(self.batter_ids[idx]),
            "game_date": self.game_dates.iloc[idx],
            "situational": self.situational[idx],
            "base_state": self.base_state[idx],
            "matchup_index": self.matchup_index[idx],
            "park_index": self.park_index[idx],
            "league_rates": self.league_rates[idx],
            "contact_quality": self.contact_quality[idx],
            "contact_quality_aux_target": self.contact_quality_aux_target[idx],
            "target": self.target[idx],
        }


class EventBatchCollator:
    """Turns a list of EventDataset samples into one training batch. The
    embedding cache lookup is the only work actually done here -- everything
    else was already precomputed per-row by EventDataset.__init__ and just
    needs stacking. pitcher_cache/batter_cache are read-only
    (src.data.event_embedding_cache.EmbeddingCache): every (player_id,
    game_date) pair this collator will ever ask for must already be in the
    precomputed cache, or .get() raises rather than silently recomputing."""

    def __init__(self, pitcher_cache: EmbeddingCache, batter_cache: EmbeddingCache) -> None:
        self.pitcher_cache = pitcher_cache
        self.batter_cache = batter_cache

    def __call__(self, batch: list[dict]) -> dict:
        game_dates = pd.Series([s["game_date"] for s in batch])
        pitcher_embedding = self.pitcher_cache.get_batch(pd.Series([s["pitcher_id"] for s in batch]), game_dates)
        batter_embedding = self.batter_cache.get_batch(pd.Series([s["batter_id"] for s in batch]), game_dates)

        context = torch.cat(
            [
                torch.stack([s["situational"] for s in batch]),
                torch.stack([s["base_state"] for s in batch]),
                torch.stack([s["league_rates"] for s in batch]),
                torch.stack([s["contact_quality"] for s in batch]),
            ],
            dim=-1,
        )

        return {
            "pitcher_embedding": pitcher_embedding,
            "batter_embedding": batter_embedding,
            "context": context,
            "contact_quality_aux_target": torch.stack([s["contact_quality_aux_target"] for s in batch]),
            "matchup_index": torch.stack([s["matchup_index"] for s in batch]),
            "park_index": torch.stack([s["park_index"] for s in batch]),
            "target": torch.stack([s["target"] for s in batch]),
        }
