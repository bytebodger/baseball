"""Combines a pitch's pitcher/batter long-history embeddings (precomputed by
src/data/event_embedding_cache.py from the frozen LongHistoryEncoder -- see
that module's docstring for why those embeddings are precomputed rather than
recomputed live here) with situational context, park factor, and
league-rate features (src/data/event_dataset.py) into a distribution over
src/data/sequence_dataset.py's OUTCOME_VOCAB.

Unlike GamePredictor, the player encoder isn't a submodule here: its
embeddings arrive as plain input tensors, already computed once and cached
to disk (~40 minutes for the full dataset's 804,536 distinct (player, date)
pairs -- see event_embedding_cache.py's docstring), so this model only ever
trains the lightweight combiner on top of them. There's deliberately no
fine-tuning stage that unfreezes the encoder the way GamePredictor's
training script can: doing so would mean recomputing the encoder's forward
pass every epoch, exactly the cost precomputing the cache was built to
avoid.

EventModelConfig.include_context toggles an ablation: when False,
ParkFactorEmbedding and the matchup embedding aren't even constructed as
submodules, and forward() concatenates only the two player embeddings into
the trunk. That's a hard architectural ablation -- the model has no
parameters connected to situational/park/league information at all, not
just a zeroed-out input path to them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.data.park_factors import DEFAULT_EMBEDDING_DIM, ParkFactorEmbedding
from src.data.sequence_dataset import MATCHUP_VOCAB, OUTCOME_VOCAB

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "event_model.yaml"


@dataclass
class EventModelConfig:
    player_embed_dim: int = 128
    matchup_embed_dim: int = 8
    park_factor_embed_dim: int = DEFAULT_EMBEDDING_DIM
    # Width of the collator's "context" tensor (situational + base-state +
    # league-rate scalars) -- see src/data/event_dataset.py's CONTEXT_DIM.
    # Kept as a plain int here (not imported from event_dataset) the same
    # way GamePredictorConfig.context_dim is just a number GamePredictor
    # trusts its training script to get right, rather than a cross-import
    # from the dataset module.
    situational_dim: int = 11
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    include_context: bool = True

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "EventModelConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


class EventModel(nn.Module):
    def __init__(
        self,
        config: EventModelConfig | None = None,
        park_factor_embedding: ParkFactorEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EventModelConfig()
        if self.config.include_context and park_factor_embedding is None:
            raise ValueError("park_factor_embedding is required when include_context=True")

        combined_dim = self.config.player_embed_dim * 2
        if self.config.include_context:
            self.park_factor_embedding = park_factor_embedding
            self.matchup_embed = nn.Embedding(len(MATCHUP_VOCAB), self.config.matchup_embed_dim)
            combined_dim += self.config.park_factor_embed_dim + self.config.matchup_embed_dim + self.config.situational_dim
        else:
            self.park_factor_embedding = None
            self.matchup_embed = None

        layers: list[nn.Module] = []
        in_dim = combined_dim
        for _ in range(self.config.num_layers):
            layers += [nn.Linear(in_dim, self.config.hidden_dim), nn.ReLU(), nn.Dropout(self.config.dropout)]
            in_dim = self.config.hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.output_head = nn.Linear(self.config.hidden_dim, len(OUTCOME_VOCAB))

    @classmethod
    def from_yaml(
        cls, park_factor_embedding: ParkFactorEmbedding | None = None, path: Path = DEFAULT_CONFIG_PATH
    ) -> "EventModel":
        return cls(EventModelConfig.from_yaml(path), park_factor_embedding)

    def forward(self, batch: dict) -> torch.Tensor:
        """batch: the dict produced by EventBatchCollator (see
        src/data/event_dataset.py). Returns raw logits over OUTCOME_VOCAB,
        shape [batch, len(OUTCOME_VOCAB)]."""
        parts = [batch["pitcher_embedding"], batch["batter_embedding"]]
        if self.config.include_context:
            park_embedding = self.park_factor_embedding(batch["park_index"])
            matchup_embedding = self.matchup_embed(batch["matchup_index"])
            parts += [park_embedding, matchup_embedding, batch["context"]]
        combined = torch.cat(parts, dim=-1)
        hidden = self.trunk(combined)
        return self.output_head(hidden)
