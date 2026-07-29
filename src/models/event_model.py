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

The contact-quality features (src/data/contact_quality.py -- pitcher exit-
velo/hard-hit-rate allowed, batter exit-velo/hard-hit-rate produced) are
passed in as 4 raw scalars folded into the same "context" tensor as the
situational/base-state/league-rate scalars (src/data/event_dataset.py's
CONTEXT_DIM includes them). An earlier version of this module routed them
through a dedicated 2-layer MLP sub-network instead, on the hypothesis that
4 bare numbers competing against two 128-dim player embeddings for gradient
descent's attention were structurally underweighted -- a
feature-ablation/gradient-sensitivity check did find smaller pitcher-side
gradients than batter-side/situational ones, and the dedicated sub-network
did increase some of those gradients, but it *decreased* the feature's
overall ablation effect on the model's output and made the 35-game
low-scoring-game calibration check (see the simulator's own validation
history) regress most of the way back to having no contact-quality feature
at all. Reverted back to this simpler raw-scalar version, which measurably
outperformed it.

contact_quality_aux_head is a small auxiliary regression head branching off
the trunk's own final hidden state -- the same representation output_head
reads from, not a separate side-branch -- predicting real, leak-safe
strictly-prior pitcher BABIP-allowed and hard-hit-rate-allowed
(src/data/contact_quality.py's babip_for/contact_quality_features_for).
train_event_model.py adds a small-weight MSE loss on this head's output
alongside the main 13-way cross-entropy loss: since it shares the trunk's
own parameters with the main classification path, gradients from this
auxiliary task push the SAME hidden representation the main task uses to
actually encode contact-quality-relevant information, rather than just
adding capacity for the raw input scalars to sit in (which is what the
now-reverted dedicated-encoder attempt did, and which measurably didn't
help). Present whenever include_context=True (contact-quality features only
exist in that mode); forward()'s `return_aux` flag defaults to False so
every existing inference-time caller (game_engine.py) is unaffected -- only
train_event_model.py opts into reading the auxiliary predictions.

base_occupancy_aux_head (2026-07-28 addition) is the same shared-trunk
auxiliary-regression mechanism as contact_quality_aux_head above, but
targeting a different, later-diagnosed gap: Phase 11 boundary-2 validation
found the model's predicted hit-probability/XBH-probability barely moves
with base occupancy (empty vs. loaded bases), while real conditional rates
rise sharply -- the base_state feature is present and correctly signed, just
severely underweighted. A first fix attempt (EventDataset.base_state_scale,
simple input scaling) tested clean methodologically but showed no
seed-consistent improvement -- plausibly because scaling a raw input by a
constant is close to a no-op under Adam's per-parameter adaptive step sizes,
which can just learn a proportionally smaller weight to compensate. This
head is the second attempt: it predicts real, train-only
(hit_rate_given_base_state, xbh_rate_given_base_state) population rates
(src.data.event_dataset.compute_base_occupancy_rates) from the same trunk
hidden state the main classification path uses, on the same theory that
made contact_quality_aux_head's approach partially work -- pushing gradient
pressure into the shared representation itself rather than the raw input's
scale, which the optimizer can't as easily undo.

EventModelConfig.interaction_type (2026-07-18 addition) selects an explicit
pitcher-batter interaction mechanism, addressing a *different* diagnosed gap
than the contact-quality one above: a matchup-interaction probe found real
2023 data shows a genuine interaction effect (batter quality matters +0.074
wOBA facing an elite pitcher vs. +0.116 facing a weak one -- aces compress
hitter quality) that the plain concatenation-plus-MLP trunk wasn't
reproducing at all (model showed +0.063 vs. +0.062 -- statistically flat,
purely additive). Concatenation feeding a generic MLP *can* represent a
cross-term in principle, but nothing pushes it to find one over the easier
additive solution, so this makes an interaction available to the model by
construction instead of hoping gradient descent discovers one on its own:
  - "none" (default): no change from the original architecture -- pitcher
    and batter embeddings are just concatenated.
  - "bilinear": a low-rank factorized bilinear term. Both embeddings are
    each linearly projected down to `interaction_dim`, then combined by
    elementwise product (the standard low-rank bilinear/factorization-
    machine trick: sum_r (pitcher @ W1)_r * (batter @ W2)_r approximates a
    full pitcher^T W batter bilinear form far more cheaply than the
    O(player_embed_dim^2) parameter cost of a literal bilinear layer). This
    interaction vector is concatenated alongside the two raw embeddings,
    not in place of them.
  - "film": FiLM-style conditioning (Perez et al., 2017) -- the pitcher
    embedding produces a per-dimension scale and shift that modulates the
    batter embedding directly: batter_modulated = (1 + scale(pitcher)) *
    batter + shift(pitcher). This *replaces* the raw batter embedding in
    the trunk's input with the pitcher-conditioned version, so pitcher
    quality mechanically compresses or amplifies the batter signal before
    the trunk ever sees it, rather than the trunk having to learn to do
    that compression itself. scale/shift are zero-initialized so the model
    starts at the identity (batter_modulated == batter at init) and only
    learns to deviate from pure additivity if the training signal rewards
    it.
  - "elementwise": the cheapest option -- just the raw elementwise product
    pitcher * batter (same dim as either embedding), concatenated alongside
    the two raw embeddings. No new learned parameters of its own; relies on
    the trunk's first linear layer to weight the product term usefully.
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


VALID_INTERACTION_TYPES = ("none", "bilinear", "film", "elementwise")


@dataclass
class EventModelConfig:
    player_embed_dim: int = 128
    matchup_embed_dim: int = 8
    park_factor_embed_dim: int = DEFAULT_EMBEDDING_DIM
    # Width of the collator's "context" tensor (situational + base-state +
    # league-rate + contact-quality scalars) -- see src/data/event_dataset.py's
    # CONTEXT_DIM. Kept as a plain int here (not imported from event_dataset)
    # the same way GamePredictorConfig.context_dim is just a number
    # GamePredictor trusts its training script to get right, rather than a
    # cross-import from the dataset module.
    situational_dim: int = 11
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    include_context: bool = True
    # See module docstring: "none" reproduces the original concatenation-only
    # architecture exactly. "bilinear"/"film"/"elementwise" add an explicit
    # pitcher-batter interaction mechanism.
    interaction_type: str = "none"
    # Only used by interaction_type="bilinear" -- width of the low-rank
    # projected interaction vector concatenated into the trunk's input.
    interaction_dim: int = 32
    # Gates base_occupancy_aux_head's existence (see module docstring) --
    # False (default) reproduces every checkpoint trained before 2026-07-28
    # exactly, with no such head at all. Deliberately NOT tied to
    # include_context alone the way contact_quality_aux_head is: that head
    # existed from early in the project, so every checkpoint has always had
    # it; base_occupancy_aux_head was added later, so making it unconditional
    # on include_context would break load_state_dict for every
    # already-trained checkpoint (a real incident during the 2026-07-28
    # investigation this head was added for -- see git history around this
    # line). An explicit flag avoids that class of break for any future
    # optional head too.
    has_base_occupancy_aux_head: bool = False

    def __post_init__(self) -> None:
        if self.interaction_type not in VALID_INTERACTION_TYPES:
            raise ValueError(f"interaction_type must be one of {VALID_INTERACTION_TYPES}, got {self.interaction_type!r}")

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

        # See module docstring for what each interaction_type does. All
        # three read the same raw pitcher_embedding/batter_embedding inputs
        # forward() already has -- none of them need new data, just new
        # ways of combining what's already there.
        self.bilinear_pitcher_proj = None
        self.bilinear_batter_proj = None
        self.film_scale = None
        self.film_shift = None
        if self.config.interaction_type == "bilinear":
            self.bilinear_pitcher_proj = nn.Linear(self.config.player_embed_dim, self.config.interaction_dim, bias=False)
            self.bilinear_batter_proj = nn.Linear(self.config.player_embed_dim, self.config.interaction_dim, bias=False)
            combined_dim += self.config.interaction_dim
        elif self.config.interaction_type == "film":
            self.film_scale = nn.Linear(self.config.player_embed_dim, self.config.player_embed_dim)
            self.film_shift = nn.Linear(self.config.player_embed_dim, self.config.player_embed_dim)
            # Zero-initialized so batter_modulated == batter_embedding at
            # init (scale starts at 0 -> gamma=1+0=1, shift starts at 0) --
            # the model starts exactly at the plain-concatenation baseline
            # and only learns to deviate from additivity if doing so
            # actually helps, rather than starting from a random modulation
            # that could hurt early training.
            nn.init.zeros_(self.film_scale.weight)
            nn.init.zeros_(self.film_scale.bias)
            nn.init.zeros_(self.film_shift.weight)
            nn.init.zeros_(self.film_shift.bias)
        elif self.config.interaction_type == "elementwise":
            combined_dim += self.config.player_embed_dim

        layers: list[nn.Module] = []
        in_dim = combined_dim
        for _ in range(self.config.num_layers):
            layers += [nn.Linear(in_dim, self.config.hidden_dim), nn.ReLU(), nn.Dropout(self.config.dropout)]
            in_dim = self.config.hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.output_head = nn.Linear(self.config.hidden_dim, len(OUTCOME_VOCAB))

        # (pitcher_babip_allowed, pitcher_hard_hit_rate_allowed) -- see
        # module docstring. Only meaningful when include_context=True (that's
        # the only mode with any contact-quality signal reaching the trunk
        # at all).
        self.contact_quality_aux_head = nn.Linear(self.config.hidden_dim, 2) if self.config.include_context else None

        # (hit_rate_given_base_state, xbh_rate_given_base_state) -- second
        # re-weighting attempt at Phase 11's base-occupancy-underweighting
        # finding (base_state_scale, EventDataset's input-scaling attempt,
        # was tested and showed no seed-consistent improvement). Same
        # shared-trunk-pressure mechanism as contact_quality_aux_head, just
        # targeting src.data.event_dataset.compute_base_occupancy_rates'
        # population-level real rates instead of per-player contact quality.
        # Only meaningful when include_context=True (base_state only reaches
        # the trunk in that mode).
        self.base_occupancy_aux_head = (
            nn.Linear(self.config.hidden_dim, 2)
            if self.config.include_context and self.config.has_base_occupancy_aux_head
            else None
        )

    @classmethod
    def from_yaml(
        cls, park_factor_embedding: ParkFactorEmbedding | None = None, path: Path = DEFAULT_CONFIG_PATH
    ) -> "EventModel":
        return cls(EventModelConfig.from_yaml(path), park_factor_embedding)

    def forward(
        self, batch: dict, return_aux: bool = False, return_base_occupancy_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """batch: the dict produced by EventBatchCollator (see
        src/data/event_dataset.py). Returns raw logits over OUTCOME_VOCAB,
        shape [batch, len(OUTCOME_VOCAB)] by default. `return_aux=True` (and
        include_context=True) additionally returns contact_quality_aux_head's
        predictions ((pitcher_babip_allowed, pitcher_hard_hit_rate_allowed),
        shape [batch, 2] -- see module docstring). `return_base_occupancy_aux=True`
        (and include_context=True) additionally returns base_occupancy_aux_head's
        predictions ((hit_rate_given_base_state, xbh_rate_given_base_state),
        shape [batch, 2]). The two flags are independent: with both True, the
        return is (logits, contact_quality_aux, base_occupancy_aux); with one
        True, (logits, that_aux); with neither (the default), just logits --
        so every existing inference-time caller (game_engine.py, which passes
        neither) is unaffected; only train_event_model.py opts into either."""
        pitcher_embedding = batch["pitcher_embedding"]
        batter_embedding = batch["batter_embedding"]

        if self.config.interaction_type == "film":
            gamma = 1.0 + self.film_scale(pitcher_embedding)
            beta = self.film_shift(pitcher_embedding)
            batter_embedding = gamma * batter_embedding + beta

        parts = [pitcher_embedding, batter_embedding]

        if self.config.interaction_type == "bilinear":
            interaction = self.bilinear_pitcher_proj(pitcher_embedding) * self.bilinear_batter_proj(batter_embedding)
            parts.append(interaction)
        elif self.config.interaction_type == "elementwise":
            parts.append(pitcher_embedding * batter_embedding)

        if self.config.include_context:
            park_embedding = self.park_factor_embedding(batch["park_index"])
            matchup_embedding = self.matchup_embed(batch["matchup_index"])
            parts += [park_embedding, matchup_embedding, batch["context"]]
        combined = torch.cat(parts, dim=-1)
        hidden = self.trunk(combined)
        logits = self.output_head(hidden)

        contact_quality_aux = None
        if return_aux:
            if self.contact_quality_aux_head is None:
                raise ValueError("return_aux=True requires include_context=True (contact_quality_aux_head doesn't exist otherwise)")
            contact_quality_aux = self.contact_quality_aux_head(hidden)

        base_occupancy_aux = None
        if return_base_occupancy_aux:
            if self.base_occupancy_aux_head is None:
                raise ValueError(
                    "return_base_occupancy_aux=True requires include_context=True (base_occupancy_aux_head doesn't exist otherwise)"
                )
            base_occupancy_aux = self.base_occupancy_aux_head(hidden)

        if return_aux and return_base_occupancy_aux:
            return logits, contact_quality_aux, base_occupancy_aux
        if return_aux:
            return logits, contact_quality_aux
        if return_base_occupancy_aux:
            return logits, base_occupancy_aux
        return logits
