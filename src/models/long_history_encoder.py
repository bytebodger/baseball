"""Hierarchical player encoder for careers longer than PlayerEncoder's
200-pitch trailing window can see.

PlayerEncoder (src/models/player_encoder.py) truncates to the most recent
200 pitches -- for an everyday starter that's roughly 6-10 starts, a few
weeks. This module trades pitch-level resolution for career-level reach: it
still uses PlayerEncoder underneath (nothing about *how* to pool a pitch
sequence changes), just applies it once per calendar month instead of once
per trailing window, then adds a second level on top that pools those
per-month embeddings across a much longer history.

Two levels:

- ChunkEncoder (level 1): exactly PlayerEncoder, reused as-is -- pools one
  calendar month's worth of a player's pitches (however many games that
  turns out to span) into a single "chunk" embedding. It's the same job
  PlayerEncoder already does for a trailing pitch window, just scoped to one
  month instead of a fixed pitch count; there's no reason to reimplement the
  same CLS-token Transformer pooling twice; identical implementation IS "the
  same architecture pattern" the two levels share. (An earlier version of
  this class ran its Transformer under gradient checkpointing, from back
  when a "chunk" was one game and max_chunks was 400 -- activation memory
  across that many flattened per-batch sequences was the dominant memory
  cost then. Measured head-to-head after chunks became calendar months
  (max_chunks 36, an ~11x drop), checkpointing bought no measurable memory
  savings -- batch input tensors and optimizer state dominate at this scale
  instead -- while still costing ~18% wall-clock time to the recomputation
  on the backward pass, so it was removed.)
- CareerEncoder (level 2): takes a player's chunk embeddings in chronological
  order and runs them through a second CLS-token Transformer, the same
  pooling recipe one level up. Capped at 36 chunks (roughly three years of
  monthly activity) rather than PlayerEncoder's 200 pitches, since a "chunk"
  here is a calendar month, not a single pitch.

The one real architectural difference from PlayerEncoder is what breaks the
symmetry of a sequence of months: real calendar gaps, not just how many
chunks apart two months are. A pitcher whose last two active months were
back-to-back should look different from one coming off a multi-month IL
stint, even though both are "the previous chunk." PlayerEncoder's
position_embed is a lookup table over integer pitch-index-in-sequence,
which can't represent that -- so CareerEncoder replaces it with
ChunkTimeEncoding, a sinusoidal encoding of each chunk's actual elapsed time
(days before the prediction cutoff) rather than its list position, the same
sin/cos frequency-bank construction the original Transformer positional
encoding uses, generalized from integer positions to continuous real-valued
ones.

LongHistoryEncoder wires the two levels together: given a batch of players'
full (player, chunk, pitch) nested history, it flattens the player/chunk
dimensions to run ChunkEncoder once over every chunk in the batch (as cheap
as encoding that many independent chunks would be), then reshapes back to
run CareerEncoder per player.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.models.player_encoder import PlayerEncoder, PlayerEncoderConfig

DEFAULT_CAREER_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "career_encoder.yaml"

DEFAULT_MAX_CHUNKS = 36


class ChunkEncoder(PlayerEncoder):
    """Level 1: pools one calendar month's worth of a player's pitches into
    a single chunk embedding. Identical architecture and forward signature
    to PlayerEncoder (see module docstring for why) -- this subclass exists
    so the hierarchical encoder's two levels each have a name that matches
    the role they play here (a plain PlayerEncoder alias would work just as
    well architecturally, but "ChunkEncoder" is what LongHistoryEncoder and
    its callers actually mean). Shares PlayerEncoderConfig and
    configs/player_encoder.yaml: the per-pitch hyperparameters (hidden size,
    heads, dropout, ...) are the same encoder doing the same job, just
    scoped to one month's pitches rather than a trailing multi-month window.
    """


ChunkEncoderConfig = PlayerEncoderConfig


class ChunkTimeEncoding(nn.Module):
    """Sinusoidal positional encoding parameterized by a continuous elapsed-
    time value -- days before the prediction cutoff -- rather than a
    discrete chunk index, so two players whose 5th-most-recent game
    happened 6 days ago vs. 6 months ago (one just back from an injury) get
    different encodings, not the same one just because both are "5th in the
    list." Same sin/cos frequency-bank construction as the original
    Transformer's positional encoding; the only change is feeding it a
    real-valued time gap instead of an integer sequence position.

    Not learned (registered as a buffer, not a parameter) -- same as the
    original Transformer's sinusoidal encoding, on the logic that a fixed,
    smooth multi-frequency basis over elapsed time generalizes to gaps the
    model never saw in training better than a learned lookup table would.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError(f"hidden_size must be even for sin/cos pairing, got {hidden_size}")
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size))
        self.register_buffer("div_term", div_term, persistent=False)

    def forward(self, days_before_cutoff: torch.Tensor) -> torch.Tensor:
        """days_before_cutoff: [batch, num_chunks] float (0 = most recent
        chunk, larger = further in the past). Returns [batch, num_chunks, hidden_size]."""
        angles = days_before_cutoff.unsqueeze(-1).float() * self.div_term
        pe = torch.zeros(*days_before_cutoff.shape, self.div_term.numel() * 2, device=days_before_cutoff.device)
        pe[..., 0::2] = torch.sin(angles)
        pe[..., 1::2] = torch.cos(angles)
        return pe


@dataclass
class CareerEncoderConfig:
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    feedforward_dim: int = 512
    # A generous cap, not a hard architectural limit like PlayerEncoder's
    # max_seq_len -- ChunkTimeEncoding is continuous, so it doesn't need a
    # lookup table sized to this. Only used by callers deciding how far back
    # to look when building a player's chunk sequence.
    max_chunks: int = DEFAULT_MAX_CHUNKS

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CAREER_CONFIG_PATH) -> "CareerEncoderConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


class CareerEncoder(nn.Module):
    """Level 2: pools a player's chronological sequence of chunk embeddings
    (one per calendar month, from ChunkEncoder) into a single career
    representation. Same CLS-token Transformer pooling PlayerEncoder uses,
    and the same zero-chunk fallback (a learned no_history_embedding, never
    run through attention) for a player with no games at all yet -- the
    career-level equivalent of PlayerEncoder's own no-history case for a
    player's first career pitch.
    """

    def __init__(self, config: CareerEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or CareerEncoderConfig()
        if self.config.hidden_size % self.config.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.config.hidden_size}) must be divisible by "
                f"num_heads ({self.config.num_heads})"
            )

        hidden_size = self.config.hidden_size

        self.time_encoding = ChunkTimeEncoding(hidden_size)
        self.input_dropout = nn.Dropout(self.config.dropout)

        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.no_history_embedding = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.no_history_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)
        self.output_norm = nn.LayerNorm(hidden_size)

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CAREER_CONFIG_PATH) -> "CareerEncoder":
        return cls(CareerEncoderConfig.from_yaml(path))

    def forward(
        self,
        chunk_embeddings: torch.Tensor,
        days_before_cutoff: torch.Tensor,
        padding_mask: torch.Tensor,
        has_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        chunk_embeddings: [batch, num_chunks, hidden_size] -- one per game,
            chronological order, from ChunkEncoder (padding chunks may hold
            any finite value; they're masked out via `padding_mask`, not by
            their content).
        days_before_cutoff: [batch, num_chunks] (float) -- days between the
            prediction cutoff and this chunk's game date; 0 = most recent
            real chunk. Values at padding positions are ignored (zeroed
            internally before use, so callers don't need to pre-clean them).
        padding_mask: [batch, num_chunks] (bool), True at padding chunks.
        has_history: [batch] (bool) -- whether this player has *any* chunk
            at all (false only for a total cold start, e.g. an MLB debut).

        Returns: [batch, hidden_size]
        """
        history_idx = has_history.nonzero(as_tuple=True)[0]
        no_history_idx = (~has_history).nonzero(as_tuple=True)[0]

        # Built by concatenating each group's output and gathering back into
        # the original batch order, rather than in-place index assignment,
        # so gradients flow cleanly through both branches -- same pattern as
        # PlayerEncoder.forward.
        parts = []
        order = []

        if no_history_idx.numel() > 0:
            no_history_out = self.no_history_embedding.unsqueeze(0).expand(no_history_idx.numel(), -1)
            parts.append(no_history_out)
            order.append(no_history_idx)

        if history_idx.numel() > 0:
            pooled = self._encode(
                chunk_embeddings[history_idx],
                days_before_cutoff[history_idx],
                padding_mask[history_idx],
            )
            parts.append(pooled)
            order.append(history_idx)

        output = torch.cat(parts, dim=0)
        order = torch.cat(order, dim=0)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.size(0), device=order.device)
        return output[inverse]

    def _encode(
        self,
        chunk_embeddings: torch.Tensor,
        days_before_cutoff: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Padding chunks' time values are never meaningful -- zero them
        # before the sin/cos encoding rather than trusting callers to have
        # done it, since a NaN or otherwise non-finite value here would
        # still poison the weighted sum in attention even though its
        # attention weight is masked to (near) zero (0 * NaN = NaN).
        days_before_cutoff = torch.where(padding_mask, torch.zeros_like(days_before_cutoff), days_before_cutoff)

        x = chunk_embeddings + self.time_encoding(days_before_cutoff)
        x = self.input_dropout(x)

        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # CLS is never padding, so it always attends and is always attended to.
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)

        encoded = self.transformer(x, src_key_padding_mask=full_mask)
        return self.output_norm(encoded[:, 0, :])


class LongHistoryEncoder(nn.Module):
    """Combines ChunkEncoder and CareerEncoder into the two-level
    hierarchical encoder this module is named for: one calendar month's
    pitches -> a chunk embedding (level 1), a chronological sequence of
    chunk embeddings -> one player representation (level 2).

    Runs ChunkEncoder once over every (player, chunk) pair in the batch,
    flattened -- exactly as cheap as encoding that many independent chunks
    would be -- then reshapes back to [batch, max_chunks, hidden] to run
    CareerEncoder over each player's own chronological chunk sequence.
    """

    def __init__(self, chunk_encoder: ChunkEncoder, career_encoder: CareerEncoder) -> None:
        super().__init__()
        if chunk_encoder.config.hidden_size != career_encoder.config.hidden_size:
            raise ValueError(
                f"chunk_encoder.hidden_size ({chunk_encoder.config.hidden_size}) must match "
                f"career_encoder.hidden_size ({career_encoder.config.hidden_size}) -- chunk "
                "embeddings are career_encoder's input."
            )
        self.chunk_encoder = chunk_encoder
        self.career_encoder = career_encoder

    def forward(
        self,
        chunk_pitch_sequences: dict[str, torch.Tensor],
        days_before_cutoff: torch.Tensor,
        chunk_padding_mask: torch.Tensor,
        has_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        chunk_pitch_sequences: dict with continuous, pitch_type, outcome,
            matchup, position, padding_mask, has_history -- the same keys
            ChunkEncoder.forward expects, but each shaped
            [batch, max_chunks, max_pitch_len, ...] (one extra max_chunks
            dimension versus a single-chunk PlayerEncoder call). Its
            "has_history" is per (player, chunk): whether that chunk slot
            has any pitches at all -- false for padding chunks beyond a
            player's real chunk count, which is exactly what makes them
            resolve to ChunkEncoder's finite learned no-history embedding
            rather than needing separate handling here.
        days_before_cutoff: [batch, max_chunks] (float), see CareerEncoder.forward.
        chunk_padding_mask: [batch, max_chunks] (bool), True at padding chunks.
        has_history: [batch] (bool), whether this player has any chunk at all.

        Returns: [batch, hidden_size]
        """
        batch_size, max_chunks = chunk_padding_mask.shape
        flat = {key: value.reshape(batch_size * max_chunks, *value.shape[2:]) for key, value in chunk_pitch_sequences.items()}
        flat_chunk_embeddings = self.chunk_encoder(**flat)
        chunk_embeddings = flat_chunk_embeddings.view(batch_size, max_chunks, -1)

        return self.career_encoder(chunk_embeddings, days_before_cutoff, chunk_padding_mask, has_history)
