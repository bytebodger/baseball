import pytest
import torch

from src.data.sequence_dataset import OUTCOME_VOCAB
from src.simulation.xbh_calibration import (
    BIP_OUTCOMES,
    XBH_OUTCOMES,
    apply_xbh_calibration_correction,
    apply_xbh_calibration_correction_batched,
)


def _distribution(**overrides) -> dict[str, float]:
    base = {o: 0.0 for o in OUTCOME_VOCAB}
    base.update(
        {
            "ball": 0.30, "called_strike": 0.15, "foul": 0.10, "swinging_strike": 0.10,
            "single": 0.10, "double": 0.05, "triple": 0.01, "home_run": 0.02, "hit_into_play_out": 0.12,
            "strikeout": 0.08, "walk": 0.05, "hit_by_pitch": 0.02,
        }
    )
    base.update(overrides)
    return base


def test_zero_gain_is_a_no_op():
    dist = _distribution()
    result = apply_xbh_calibration_correction(dist, real_xbh_rate_of_bip=0.5, gain=0.0)
    assert result == dist


def test_correction_preserves_total_probability_and_p_bip():
    dist = _distribution()
    p_bip_before = sum(dist[o] for o in BIP_OUTCOMES)
    result = apply_xbh_calibration_correction(dist, real_xbh_rate_of_bip=0.5, gain=1.0)
    assert sum(result.values()) == pytest.approx(sum(dist.values()))
    assert sum(result[o] for o in BIP_OUTCOMES) == pytest.approx(p_bip_before)


def test_correction_never_touches_non_bip_categories():
    dist = _distribution()
    result = apply_xbh_calibration_correction(dist, real_xbh_rate_of_bip=0.9, gain=1.0)
    for o in ["ball", "called_strike", "foul", "swinging_strike", "strikeout", "walk", "hit_by_pitch"]:
        assert result[o] == pytest.approx(dist[o])


def test_full_gain_matches_real_rate_exactly():
    dist = _distribution()
    p_bip = sum(dist[o] for o in BIP_OUTCOMES)
    result = apply_xbh_calibration_correction(dist, real_xbh_rate_of_bip=0.5, gain=1.0)
    corrected_share = sum(result[o] for o in XBH_OUTCOMES) / sum(result[o] for o in BIP_OUTCOMES)
    assert corrected_share == pytest.approx(0.5, abs=1e-6)
    assert sum(result[o] for o in BIP_OUTCOMES) == pytest.approx(p_bip)


def test_partial_gain_moves_partway_toward_real_rate():
    dist = _distribution()
    model_share = sum(dist[o] for o in XBH_OUTCOMES) / sum(dist[o] for o in BIP_OUTCOMES)
    real_rate = 0.5
    result = apply_xbh_calibration_correction(dist, real_xbh_rate_of_bip=real_rate, gain=0.5)
    corrected_share = sum(result[o] for o in XBH_OUTCOMES) / sum(result[o] for o in BIP_OUTCOMES)
    expected = model_share + 0.5 * (real_rate - model_share)
    assert corrected_share == pytest.approx(expected, abs=1e-6)


def test_negative_direction_shrinks_xbh_share_when_real_rate_is_lower():
    dist = _distribution()
    model_share = sum(dist[o] for o in XBH_OUTCOMES) / sum(dist[o] for o in BIP_OUTCOMES)
    result = apply_xbh_calibration_correction(dist, real_xbh_rate_of_bip=0.01, gain=1.0)
    corrected_share = sum(result[o] for o in XBH_OUTCOMES) / sum(result[o] for o in BIP_OUTCOMES)
    assert corrected_share < model_share
    assert corrected_share == pytest.approx(0.01, abs=1e-6)


def test_relative_mix_within_xbh_and_non_xbh_groups_is_preserved():
    """The correction should scale each sub-group multiplicatively, not
    change the RELATIVE split between e.g. double/triple/home_run within
    the XBH group, or single/hit_into_play_out within the non-XBH group."""
    dist = _distribution()
    result = apply_xbh_calibration_correction(dist, real_xbh_rate_of_bip=0.5, gain=0.7)
    assert result["double"] / result["triple"] == pytest.approx(dist["double"] / dist["triple"])
    assert result["single"] / result["hit_into_play_out"] == pytest.approx(dist["single"] / dist["hit_into_play_out"])


def test_no_bip_probability_is_a_safe_no_op():
    dist = {o: 0.0 for o in OUTCOME_VOCAB}
    dist["ball"] = 1.0
    result = apply_xbh_calibration_correction(dist, real_xbh_rate_of_bip=0.5, gain=1.0)
    assert result == dist


# ---------- batched version ----------


def _batched_probs(n: int) -> torch.Tensor:
    dist = _distribution()
    row = torch.tensor([dist[o] for o in OUTCOME_VOCAB], dtype=torch.float32)
    return row.unsqueeze(0).repeat(n, 1)


def test_batched_zero_gain_is_a_no_op():
    probs = _batched_probs(4)
    real_rates = torch.tensor([0.05, 0.5, 0.9, 0.02])
    result = apply_xbh_calibration_correction_batched(probs, real_rates, gain=0.0)
    assert torch.equal(result, probs)


def test_batched_matches_single_instance_row_by_row():
    probs = _batched_probs(3)
    real_rates = torch.tensor([0.05, 0.5, 0.9])
    result = apply_xbh_calibration_correction_batched(probs, real_rates, gain=0.6)

    dist = _distribution()
    for i, real_rate in enumerate(real_rates.tolist()):
        expected = apply_xbh_calibration_correction(dist, real_xbh_rate_of_bip=real_rate, gain=0.6)
        for j, outcome in enumerate(OUTCOME_VOCAB):
            assert result[i, j].item() == pytest.approx(expected[outcome], abs=1e-5)


def test_batched_preserves_total_probability_per_row():
    probs = _batched_probs(5)
    real_rates = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])
    result = apply_xbh_calibration_correction_batched(probs, real_rates, gain=1.0)
    assert torch.allclose(result.sum(dim=1), probs.sum(dim=1), atol=1e-5)


def test_batched_different_rows_get_different_real_rates_independently():
    probs = _batched_probs(2)
    real_rates = torch.tensor([0.01, 0.99])
    result = apply_xbh_calibration_correction_batched(probs, real_rates, gain=1.0)
    from src.data.sequence_dataset import OUTCOME_INDEX

    bip = ["single", "double", "triple", "home_run", "hit_into_play_out"]
    xbh = ["double", "triple", "home_run"]
    for i, expected_share in enumerate([0.01, 0.99]):
        p_bip = sum(result[i, OUTCOME_INDEX[o]].item() for o in bip)
        p_xbh = sum(result[i, OUTCOME_INDEX[o]].item() for o in xbh)
        assert p_xbh / p_bip == pytest.approx(expected_share, abs=1e-5)
