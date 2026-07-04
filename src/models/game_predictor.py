"""Predicts a game's outcome from pre-game player and context features (see
src/data/game_dataset.py for how those features are assembled, leakage-free,
from Statcast history).

GamePredictor itself only combines already-computed embeddings:
- home/away starting pitcher embeddings, produced from that pitcher's raw
  pitch-history sequence via the shared `player_encoder` (see
  `encode_players`) -- a starter is a single player per game, so this is a
  plain padded batch, the same shape PlayerEncoder.forward already expects.
- home/away *pooled* bullpen embeddings and home/away *pooled* lineup
  embeddings, e.g. produced upstream by running each set member through
  `encode_players` and then a PlayerSetPooler (src/models/set_pooling.py).
  Encoding+pooling a whole roster is a nested variable-size-set-of-
  variable-length-sequences problem best solved once by a shared
  collate/dataset step, not duplicated inside this module -- GamePredictor
  just consumes the resulting fixed-size embedding either way.
- context_features: a pre-assembled float tensor of pre-game context (month,
  starter rest days, park/team, etc.) from the Phase 6 game table.

The player_encoder is a real submodule (not just used to size things), so
its weights are checkpointed alongside GamePredictor's and, when
`freeze_player_encoder=False`, gradients from the game outcome/runs losses
fine-tune it too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.models.player_encoder import PlayerEncoder

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "game_predictor.yaml"

_RUNS_DISTRIBUTIONS = {"negative_binomial", "regression"}


@dataclass
class GamePredictorConfig:
    context_dim: int = 8
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    # "negative_binomial": overdispersed count head (mean + total_count) --
    # a single-game run total is noisier than a Poisson would allow.
    # "regression": plain non-negative point estimate.
    runs_distribution: str = "negative_binomial"
    freeze_player_encoder: bool = False

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "GamePredictorConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


class NegativeBinomialHead(nn.Module):
    """mean (mu) + total_count (r) parameterization: variance = mu + mu^2/r,
    so a smaller total_count means more overdispersion than Poisson allows."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mean_proj = nn.Linear(hidden_dim, 1)
        self.total_count_proj = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        mean = F.softplus(self.mean_proj(hidden).squeeze(-1)) + 1e-3
        total_count = F.softplus(self.total_count_proj(hidden).squeeze(-1)) + 1e-3
        return {"mean": mean, "total_count": total_count}

    @staticmethod
    def to_distribution(mean: torch.Tensor, total_count: torch.Tensor) -> torch.distributions.NegativeBinomial:
        probs = mean / (mean + total_count)
        return torch.distributions.NegativeBinomial(total_count=total_count, probs=probs)


class RegressionRunsHead(nn.Module):
    """Plain non-negative point estimate of runs scored."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"mean": F.softplus(self.proj(hidden).squeeze(-1))}


class GamePredictor(nn.Module):
    def __init__(self, player_encoder: PlayerEncoder, config: GamePredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or GamePredictorConfig()
        if self.config.runs_distribution not in _RUNS_DISTRIBUTIONS:
            raise ValueError(
                f"runs_distribution must be one of {_RUNS_DISTRIBUTIONS}, got {self.config.runs_distribution!r}"
            )

        self.player_encoder = player_encoder
        if self.config.freeze_player_encoder:
            for param in self.player_encoder.parameters():
                param.requires_grad_(False)
            self.player_encoder.eval()

        embed_dim = self.player_encoder.config.hidden_size
        # 6 player-embedding groups: home/away starter, home/away bullpen, home/away lineup.
        combined_dim = embed_dim * 6 + self.config.context_dim

        layers: list[nn.Module] = []
        in_dim = combined_dim
        for _ in range(self.config.num_layers):
            layers += [nn.Linear(in_dim, self.config.hidden_dim), nn.ReLU(), nn.Dropout(self.config.dropout)]
            in_dim = self.config.hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.win_prob_head = nn.Linear(self.config.hidden_dim, 1)

        runs_head_cls = NegativeBinomialHead if self.config.runs_distribution == "negative_binomial" else RegressionRunsHead
        self.home_runs_head = runs_head_cls(self.config.hidden_dim)
        self.away_runs_head = runs_head_cls(self.config.hidden_dim)

    @classmethod
    def from_yaml(cls, player_encoder: PlayerEncoder, path: Path = DEFAULT_CONFIG_PATH) -> "GamePredictor":
        return cls(player_encoder, GamePredictorConfig.from_yaml(path))

    def train(self, mode: bool = True) -> "GamePredictor":
        super().train(mode)
        if self.config.freeze_player_encoder:
            self.player_encoder.eval()
        return self

    def encode_players(self, sequences: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run a padded batch of player pitch-history sequences through the
        shared player encoder. `sequences` holds exactly the kwargs
        PlayerEncoder.forward expects (continuous, pitch_type, outcome,
        matchup, position, padding_mask, has_history). Used for starters
        here; reusable upstream for encoding bullpen/lineup members before
        they're pooled into the embeddings `forward` expects.
        """
        return self.player_encoder(**sequences)

    def forward(
        self,
        home_starter_sequence: dict[str, torch.Tensor],
        away_starter_sequence: dict[str, torch.Tensor],
        home_bullpen_embedding: torch.Tensor,
        away_bullpen_embedding: torch.Tensor,
        home_lineup_embedding: torch.Tensor,
        away_lineup_embedding: torch.Tensor,
        context_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        home_starter_sequence / away_starter_sequence: dict of padded-batch
            tensors, see `encode_players`.
        home_bullpen_embedding / away_bullpen_embedding: [batch, embed_dim],
            already pooled across that team's available relievers (or the
            pooler's learned empty-set embedding, for a short/empty bullpen).
        home_lineup_embedding / away_lineup_embedding: [batch, embed_dim],
            already pooled across that team's lineup (or the pooler's
            empty-set embedding, for an incomplete lineup).
        context_features: [batch, context_dim] float tensor of pre-game
            context (month, starter rest days, park/team, etc.).

        Returns a dict with win_logit/win_prob and each team's predicted-runs
        head outputs (a "mean" key, plus "total_count" for the
        negative-binomial head).
        """
        home_starter_embedding = self.encode_players(home_starter_sequence)
        away_starter_embedding = self.encode_players(away_starter_sequence)

        combined = torch.cat(
            [
                home_starter_embedding,
                away_starter_embedding,
                home_bullpen_embedding,
                away_bullpen_embedding,
                home_lineup_embedding,
                away_lineup_embedding,
                context_features,
            ],
            dim=-1,
        )
        hidden = self.trunk(combined)

        win_logit = self.win_prob_head(hidden).squeeze(-1)
        home_runs = self.home_runs_head(hidden)
        away_runs = self.away_runs_head(hidden)

        return {
            "win_logit": win_logit,
            "win_prob": torch.sigmoid(win_logit),
            "home_runs": home_runs,
            "away_runs": away_runs,
        }
