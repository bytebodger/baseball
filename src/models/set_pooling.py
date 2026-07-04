"""Attention-based pooling of a variable-size set of player embeddings (a
bullpen's arms or a lineup's spots) into a single fixed-size vector.

Unlike mean/sum pooling, a learned query attends over the set so the model
can weight some players more heavily than others (e.g. a closer vs. a
mop-up reliever). A short bullpen or an incomplete lineup is handled via a
padding mask; a completely empty set (zero players) never goes through
attention -- it gets one learned "empty set" embedding instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "set_pooling.yaml"


@dataclass
class PlayerSetPoolerConfig:
    embed_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.1

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "PlayerSetPoolerConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


class PlayerSetPooler(nn.Module):
    def __init__(self, config: PlayerSetPoolerConfig | None = None) -> None:
        super().__init__()
        self.config = config or PlayerSetPoolerConfig()
        if self.config.embed_dim % self.config.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.config.embed_dim}) must be divisible by "
                f"num_heads ({self.config.num_heads})"
            )

        embed_dim = self.config.embed_dim

        self.query = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.empty_set_embedding = nn.Parameter(torch.empty(embed_dim))
        nn.init.normal_(self.query, std=0.02)
        nn.init.normal_(self.empty_set_embedding, std=0.02)

        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "PlayerSetPooler":
        return cls(PlayerSetPoolerConfig.from_yaml(path))

    @staticmethod
    def pad_embeddings(embeddings: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a padded batch + padding_mask from a list of variable-size
        per-item embedding sets.

        embeddings[i]: [n_i, embed_dim], one row per player in that batch
            item's set (a bullpen or a lineup). n_i may be 0 (e.g. an empty
            bullpen or a lineup slot with no player yet).

        Returns (padded, padding_mask):
            padded: [batch, max_set_size, embed_dim], zero-padded past n_i.
            padding_mask: [batch, max_set_size] bool, True at padded slots.
        """
        batch_size = len(embeddings)
        embed_dim = embeddings[0].size(-1)
        set_sizes = [e.size(0) for e in embeddings]
        max_set_size = max(max(set_sizes), 1)

        padded = embeddings[0].new_zeros(batch_size, max_set_size, embed_dim)
        padding_mask = torch.ones(batch_size, max_set_size, dtype=torch.bool, device=embeddings[0].device)
        for i, e in enumerate(embeddings):
            n = e.size(0)
            if n > 0:
                padded[i, :n] = e
                padding_mask[i, :n] = False

        return padded, padding_mask

    def forward(self, embeddings: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        embeddings: [batch, max_set_size, embed_dim], zero-padded past each
            batch item's actual set size.
        padding_mask: [batch, max_set_size] bool, True at padded (non-existent)
            slots. None means every slot holds a real player.

        Returns: [batch, embed_dim]
        """
        batch_size, max_set_size, _ = embeddings.shape
        if padding_mask is None:
            padding_mask = torch.zeros(batch_size, max_set_size, dtype=torch.bool, device=embeddings.device)

        set_sizes = (~padding_mask).sum(dim=1)
        empty_idx = (set_sizes == 0).nonzero(as_tuple=True)[0]
        nonempty_idx = (set_sizes > 0).nonzero(as_tuple=True)[0]

        # Built by concatenating each group's output and gathering back into
        # the original batch order, rather than in-place index assignment,
        # so gradients flow cleanly through both branches.
        parts = []
        order = []

        if empty_idx.numel() > 0:
            empty_out = self.empty_set_embedding.unsqueeze(0).expand(empty_idx.numel(), -1)
            parts.append(empty_out)
            order.append(empty_idx)

        if nonempty_idx.numel() > 0:
            pooled = self._pool(embeddings[nonempty_idx], padding_mask[nonempty_idx])
            parts.append(pooled)
            order.append(nonempty_idx)

        output = torch.cat(parts, dim=0)
        order = torch.cat(order, dim=0)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.size(0), device=order.device)
        return output[inverse]

    def _pool(self, embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size = embeddings.size(0)
        query = self.query.expand(batch_size, -1, -1)
        pooled, _ = self.attention(query, embeddings, embeddings, key_padding_mask=padding_mask)
        return self.output_norm(pooled.squeeze(1))
