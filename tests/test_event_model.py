import pandas as pd
import pytest
import torch

from src.data.park_factors import ParkFactorConfig, ParkFactorEmbedding, compute_park_factors
from src.data.sequence_dataset import MATCHUP_VOCAB, OUTCOME_VOCAB
from src.models.event_model import EventModel, EventModelConfig


def _tiny_park_factor_embedding() -> ParkFactorEmbedding:
    pitches = pd.DataFrame(
        {
            "game_pk": [1, 2, 3, 4],
            "park_id": ["PARK_A", "PARK_A", "PARK_A", "PARK_A"],
            "season": [2022, 2022, 2023, 2023],
            "home_score": [3, 4, 2, 5],
            "away_score": [2, 1, 3, 0],
            "outcome": ["home_run", "single", "ball", "home_run"],
        }
    )
    config = ParkFactorConfig(rolling_years=3, embedding_dim=4)
    park_factors = compute_park_factors(pitches, rolling_years=config.rolling_years)
    return ParkFactorEmbedding(config, park_factors)


def _make_batch(batch_size: int, player_embed_dim: int, situational_dim: int) -> dict:
    return {
        "pitcher_embedding": torch.randn(batch_size, player_embed_dim),
        "batter_embedding": torch.randn(batch_size, player_embed_dim),
        "context": torch.randn(batch_size, situational_dim),
        "matchup_index": torch.randint(0, len(MATCHUP_VOCAB), (batch_size,)),
        "park_index": torch.zeros(batch_size, dtype=torch.long),
        "target": torch.randint(0, len(OUTCOME_VOCAB), (batch_size,)),
    }


def test_event_model_config_from_yaml_loads_the_real_config_file():
    config = EventModelConfig.from_yaml()
    assert config.player_embed_dim > 0
    assert config.include_context is True


def test_event_model_requires_park_factor_embedding_when_include_context_true():
    config = EventModelConfig(player_embed_dim=8, situational_dim=11, include_context=True)
    with pytest.raises(ValueError):
        EventModel(config, park_factor_embedding=None)


def test_event_model_forward_output_shape_with_context():
    config = EventModelConfig(player_embed_dim=8, matchup_embed_dim=4, park_factor_embed_dim=4, situational_dim=11, hidden_dim=16, num_layers=1, include_context=True)
    model = EventModel(config, _tiny_park_factor_embedding())
    batch = _make_batch(batch_size=5, player_embed_dim=8, situational_dim=11)
    logits = model(batch)
    assert logits.shape == (5, len(OUTCOME_VOCAB))


def test_event_model_forward_output_shape_without_context():
    config = EventModelConfig(player_embed_dim=8, hidden_dim=16, num_layers=1, include_context=False)
    model = EventModel(config, park_factor_embedding=None)
    batch = _make_batch(batch_size=5, player_embed_dim=8, situational_dim=11)
    logits = model(batch)
    assert logits.shape == (5, len(OUTCOME_VOCAB))


def test_event_model_without_context_has_no_park_or_matchup_submodules():
    config = EventModelConfig(player_embed_dim=8, hidden_dim=16, num_layers=1, include_context=False)
    model = EventModel(config, park_factor_embedding=_tiny_park_factor_embedding())  # even if passed, must be dropped
    assert model.park_factor_embedding is None
    assert model.matchup_embed is None


def test_event_model_without_context_output_is_invariant_to_context_inputs():
    """The ablation must be architectural, not just a zeroed input: changing
    context/matchup/park inputs should not move the output at all when
    include_context=False."""
    torch.manual_seed(0)
    config = EventModelConfig(player_embed_dim=8, hidden_dim=16, num_layers=1, include_context=False)
    model = EventModel(config, park_factor_embedding=None)
    model.eval()

    batch_a = _make_batch(batch_size=5, player_embed_dim=8, situational_dim=11)
    batch_b = dict(batch_a)
    batch_b["context"] = torch.randn_like(batch_a["context"]) * 100
    batch_b["matchup_index"] = torch.randint(0, len(MATCHUP_VOCAB), (5,))
    batch_b["park_index"] = torch.randint(0, 3, (5,))

    with torch.no_grad():
        out_a = model(batch_a)
        out_b = model(batch_b)
    assert torch.equal(out_a, out_b)


def test_event_model_with_context_combined_dim_includes_all_pieces():
    config = EventModelConfig(player_embed_dim=8, matchup_embed_dim=4, park_factor_embed_dim=4, situational_dim=11, hidden_dim=16, num_layers=1, include_context=True)
    model = EventModel(config, _tiny_park_factor_embedding())
    expected_in = 8 * 2 + 4 + 4 + 11
    first_linear = model.trunk[0]
    assert first_linear.in_features == expected_in


def test_event_model_without_context_combined_dim_is_just_the_two_embeddings():
    config = EventModelConfig(player_embed_dim=8, hidden_dim=16, num_layers=1, include_context=False)
    model = EventModel(config, park_factor_embedding=None)
    first_linear = model.trunk[0]
    assert first_linear.in_features == 16
