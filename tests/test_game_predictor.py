import torch

from src.data.sequence_dataset import MATCHUP_VOCAB, OUTCOME_VOCAB, PITCH_TYPE_VOCAB
from src.models.game_predictor import GamePredictor, GamePredictorConfig
from src.models.player_encoder import PlayerEncoder, PlayerEncoderConfig

EMBED_DIM = 16
CONTEXT_DIM = 5


def _small_player_encoder() -> PlayerEncoder:
    return PlayerEncoder(
        PlayerEncoderConfig(hidden_size=EMBED_DIM, num_layers=1, num_heads=2, dropout=0.0, feedforward_dim=32, max_seq_len=10)
    )


def _small_predictor_config(**overrides) -> GamePredictorConfig:
    defaults = dict(context_dim=CONTEXT_DIM, hidden_dim=32, num_layers=1, dropout=0.0)
    defaults.update(overrides)
    return GamePredictorConfig(**defaults)


def _make_starter_batch(lengths: list[int]) -> dict[str, torch.Tensor]:
    """Padded starter-pitcher sequences, mimicking GameOutcomeDataset's
    home_starter/away_starter entries after collation."""
    batch_size = len(lengths)
    max_len = max(max(lengths), 1)

    continuous = torch.zeros(batch_size, max_len, 4)
    pitch_type = torch.zeros(batch_size, max_len, dtype=torch.long)
    outcome = torch.zeros(batch_size, max_len, dtype=torch.long)
    matchup = torch.zeros(batch_size, max_len, dtype=torch.long)
    position = torch.zeros(batch_size, max_len, dtype=torch.long)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    has_history = torch.zeros(batch_size, dtype=torch.bool)

    for i, length in enumerate(lengths):
        if length == 0:
            continue
        has_history[i] = True
        continuous[i, :length] = torch.randn(length, 4)
        pitch_type[i, :length] = torch.randint(0, len(PITCH_TYPE_VOCAB), (length,))
        outcome[i, :length] = torch.randint(0, len(OUTCOME_VOCAB), (length,))
        matchup[i, :length] = torch.randint(0, len(MATCHUP_VOCAB), (length,))
        position[i, :length] = torch.arange(length)
        padding_mask[i, :length] = False

    return {
        "continuous": continuous,
        "pitch_type": pitch_type,
        "outcome": outcome,
        "matchup": matchup,
        "position": position,
        "padding_mask": padding_mask,
        "has_history": has_history,
    }


def _make_forward_inputs(batch_size: int):
    home_starter = _make_starter_batch([3] * batch_size)
    away_starter = _make_starter_batch([2] * batch_size)
    home_bullpen = torch.randn(batch_size, EMBED_DIM)
    away_bullpen = torch.randn(batch_size, EMBED_DIM)
    home_lineup = torch.randn(batch_size, EMBED_DIM)
    away_lineup = torch.randn(batch_size, EMBED_DIM)
    context = torch.randn(batch_size, CONTEXT_DIM)
    return home_starter, away_starter, home_bullpen, away_bullpen, home_lineup, away_lineup, context


def test_output_shapes_and_ranges_with_negative_binomial_head():
    model = GamePredictor(_small_player_encoder(), _small_predictor_config(runs_distribution="negative_binomial"))
    model.eval()

    inputs = _make_forward_inputs(batch_size=4)
    output = model(*inputs)

    assert output["win_prob"].shape == (4,)
    assert torch.all((output["win_prob"] >= 0) & (output["win_prob"] <= 1))

    for side in ("home_runs", "away_runs"):
        assert output[side]["mean"].shape == (4,)
        assert output[side]["total_count"].shape == (4,)
        assert torch.all(output[side]["mean"] > 0)
        assert torch.all(output[side]["total_count"] > 0)


def test_negative_binomial_head_produces_a_valid_distribution():
    model = GamePredictor(_small_player_encoder(), _small_predictor_config())
    model.eval()

    inputs = _make_forward_inputs(batch_size=3)
    output = model(*inputs)

    from src.models.game_predictor import NegativeBinomialHead

    dist = NegativeBinomialHead.to_distribution(output["home_runs"]["mean"], output["home_runs"]["total_count"])
    sample = dist.sample()
    assert sample.shape == (3,)
    assert torch.all(sample >= 0)


def test_regression_runs_head_outputs_nonnegative_mean_only():
    model = GamePredictor(_small_player_encoder(), _small_predictor_config(runs_distribution="regression"))
    model.eval()

    inputs = _make_forward_inputs(batch_size=4)
    output = model(*inputs)

    assert set(output["home_runs"].keys()) == {"mean"}
    assert torch.all(output["home_runs"]["mean"] > 0)


def test_invalid_runs_distribution_raises():
    import pytest

    with pytest.raises(ValueError):
        GamePredictor(_small_player_encoder(), _small_predictor_config(runs_distribution="bogus"))


def test_frozen_player_encoder_gets_no_gradient_and_stays_in_eval():
    model = GamePredictor(_small_player_encoder(), _small_predictor_config(freeze_player_encoder=True))
    model.train()  # should NOT put the frozen player_encoder into train mode
    assert not model.player_encoder.training

    inputs = _make_forward_inputs(batch_size=4)
    output = model(*inputs)
    output["win_logit"].sum().backward()

    for param in model.player_encoder.parameters():
        assert param.grad is None
    # the combining trunk should still learn
    assert model.trunk[0].weight.grad is not None
    assert not torch.all(model.trunk[0].weight.grad == 0)


def test_finetunable_player_encoder_gets_gradients():
    model = GamePredictor(_small_player_encoder(), _small_predictor_config(freeze_player_encoder=False))
    model.train()
    assert model.player_encoder.training

    inputs = _make_forward_inputs(batch_size=4)
    output = model(*inputs)
    output["win_logit"].sum().backward()

    assert model.player_encoder.continuous_proj.weight.grad is not None
    assert not torch.all(model.player_encoder.continuous_proj.weight.grad == 0)


def test_incomplete_lineup_and_short_bullpen_run_without_crashing():
    """A pooled bullpen/lineup embedding of all zeros stands in for a
    PlayerSetPooler's learned empty-set output -- GamePredictor doesn't care
    how the embedding was produced, just that it's the right shape."""
    model = GamePredictor(_small_player_encoder(), _small_predictor_config())
    model.eval()

    home_starter, away_starter, _, away_bullpen, _, away_lineup, context = _make_forward_inputs(batch_size=2)
    empty_home_bullpen = torch.zeros(2, EMBED_DIM)
    empty_home_lineup = torch.zeros(2, EMBED_DIM)

    output = model(home_starter, away_starter, empty_home_bullpen, away_bullpen, empty_home_lineup, away_lineup, context)
    assert output["win_prob"].shape == (2,)
