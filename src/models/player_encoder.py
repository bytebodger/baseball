"""Transformer encoder that pools a player's pitch-sequence history (as
produced by PlayerPitchSequenceDataset) into a single fixed-size embedding.

Expects batches that are already padded and masked -- e.g. by a collate_fn
that stacks PlayerPitchSequenceDataset samples up to the batch's longest
sequence. Zero-history samples (has_history=False) never go through the
transformer; they get one learned "no-history" embedding instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.data.sequence_dataset import CONTINUOUS_FEATURES, MATCHUP_VOCAB, OUTCOME_VOCAB, PITCH_TYPE_VOCAB

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "player_encoder.yaml"


@dataclass
class PlayerEncoderConfig:
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    feedforward_dim: int = 512
    # Must be >= the max_seq_len used to build PlayerPitchSequenceDataset
    # sequences -- it only sizes the positional embedding table.
    max_seq_len: int = 200

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "PlayerEncoderConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


class PlayerEncoder(nn.Module):
    def __init__(self, config: PlayerEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or PlayerEncoderConfig()
        if self.config.hidden_size % self.config.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.config.hidden_size}) must be divisible by "
                f"num_heads ({self.config.num_heads})"
            )

        hidden_size = self.config.hidden_size

        self.continuous_proj = nn.Linear(len(CONTINUOUS_FEATURES), hidden_size)
        self.pitch_type_embed = nn.Embedding(len(PITCH_TYPE_VOCAB), hidden_size)
        self.outcome_embed = nn.Embedding(len(OUTCOME_VOCAB), hidden_size)
        self.matchup_embed = nn.Embedding(len(MATCHUP_VOCAB), hidden_size)
        self.position_embed = nn.Embedding(self.config.max_seq_len, hidden_size)
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
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "PlayerEncoder":
        return cls(PlayerEncoderConfig.from_yaml(path))

    def forward(
        self,
        continuous: torch.Tensor,
        pitch_type: torch.Tensor,
        outcome: torch.Tensor,
        matchup: torch.Tensor,
        position: torch.Tensor,
        padding_mask: torch.Tensor,
        has_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        continuous: [batch, seq_len, n_continuous_features] (float)
        pitch_type, outcome, matchup, position: [batch, seq_len] (long)
        padding_mask: [batch, seq_len] (bool), True at padding positions
        has_history: [batch] (bool)

        Returns: [batch, hidden_size]
        """
        history_idx = has_history.nonzero(as_tuple=True)[0]
        no_history_idx = (~has_history).nonzero(as_tuple=True)[0]

        # Built by concatenating each group's output and gathering back into
        # the original batch order, rather than in-place index assignment,
        # so gradients flow cleanly through both branches.
        parts = []
        order = []

        if no_history_idx.numel() > 0:
            no_history_out = self.no_history_embedding.unsqueeze(0).expand(no_history_idx.numel(), -1)
            parts.append(no_history_out)
            order.append(no_history_idx)

        if history_idx.numel() > 0:
            pooled = self._encode(
                continuous[history_idx],
                pitch_type[history_idx],
                outcome[history_idx],
                matchup[history_idx],
                position[history_idx],
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
        continuous: torch.Tensor,
        pitch_type: torch.Tensor,
        outcome: torch.Tensor,
        matchup: torch.Tensor,
        position: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = (
            self.continuous_proj(continuous)
            + self.pitch_type_embed(pitch_type)
            + self.outcome_embed(outcome)
            + self.matchup_embed(matchup)
            + self.position_embed(position)
        )
        x = self.input_dropout(x)

        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # CLS is never padding, so it always attends and is always attended to.
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)

        encoded = self.transformer(x, src_key_padding_mask=full_mask)
        return self.output_norm(encoded[:, 0, :])
