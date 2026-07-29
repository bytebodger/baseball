"""Simulation-layer XBH-probability calibration correction (Phase 11
boundary-2, added 2026-07-28) -- a post-hoc adjustment applied to the event
model's predicted outcome distribution during simulation, NOT a change to
the trunk itself.

Context: Phase 10's contact-quality investigation found the event model's
predicted outcome distribution GIVEN A BALL IS IN PLAY is nearly flat
regardless of real pitcher quality (predicted BABIP/XBH/HR given contact
barely differs between an ace and a replacement-level arm, despite a real,
substantial gap). Nine attempts to fix this INSIDE the trunk (reweighted
loss, the raw feature alone, a dedicated sub-network, auxiliary supervision,
weight sweeps, interaction architectures, a real feature, a noise control)
all failed or regressed the 35-game low-scoring calibration check -- see
Known Limitations item 4. A direct re-measurement (2026-07-28, in the
correct "of batted balls" units this module also uses) confirmed the same
flatness specifically for XBH-rate: correlation between real per-pitcher
XBH-rate-of-batted-balls and the model's predicted XBH share of batted
balls across 104 real 2025 starters was -0.05 (Pearson), slope -0.07 --
essentially zero relationship, and this flatness was separately found to
carry a real, substantial cost: +0.50 runs/game of the still-open
full-game-simulation elevation (see Known Limitations item 2's
expected-runs recalculation).

This module is a genuinely different kind of fix from all nine trunk-level
attempts: it doesn't retrain, doesn't touch the trunk's shared
representation, and doesn't add an input dimension, so it isn't exposed to
the general trunk-fragility failure mode that sank every one of them
(see EventModel's module docstring). Instead, it shifts probability mass
directly in the model's OUTPUT distribution, at simulation time, toward a
real, leak-safe rolling per-pitcher XBH-rate-of-batted-balls
(src.data.xbh_rate_history.XbhRateHistory) -- the same leak-safe rolling-
stat pattern already used for contact-quality's own input features, just
applied to a correction instead of a trunk input.

Mechanics: only the 5 "ball in play" terminal outcomes (single, double,
triple, home_run, hit_into_play_out) are touched. The XBH share within that
block (double+triple+home_run, as a fraction of all 5) is shifted toward the
real rate by `gain` (0=no correction, 1=fully replace the model's own share
with the real rate), and the two sub-groups (XBH vs. non-XBH-BIP) are each
rescaled multiplicatively to hit their new target totals -- this conserves
P(ball in play) exactly (never touched) and total probability exactly (every
non-BIP category, e.g. ball/strike/foul/strikeout/walk, is left completely
untouched). `gain` defaults to 0.0 (off) -- same "explicit, off-by-default,
one real attempt" convention as BaserunningConfig.two_out_aggression_boost.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.sequence_dataset import OUTCOME_INDEX

BIP_OUTCOMES = ["single", "double", "triple", "home_run", "hit_into_play_out"]
XBH_OUTCOMES = ["double", "triple", "home_run"]
NON_XBH_BIP_OUTCOMES = ["single", "hit_into_play_out"]

BIP_INDICES = [OUTCOME_INDEX[o] for o in BIP_OUTCOMES]
XBH_INDICES = [OUTCOME_INDEX[o] for o in XBH_OUTCOMES]
NON_XBH_BIP_INDICES = [OUTCOME_INDEX[o] for o in NON_XBH_BIP_OUTCOMES]


@dataclass
class XbhCalibrationConfig:
    gain: float = 0.0


def apply_xbh_calibration_correction(distribution: dict[str, float], real_xbh_rate_of_bip: float, gain: float) -> dict[str, float]:
    """Single-instance dict version (same shape as
    src.simulation.game_engine.event_outcome_distribution's return value).
    See module docstring for the mechanism."""
    if gain == 0.0:
        return dict(distribution)

    p_bip = sum(distribution.get(o, 0.0) for o in BIP_OUTCOMES)
    if p_bip <= 1e-8:
        return dict(distribution)
    p_xbh = sum(distribution.get(o, 0.0) for o in XBH_OUTCOMES)
    p_non_xbh_bip = p_bip - p_xbh

    model_share = p_xbh / p_bip
    target_share = model_share + gain * (real_xbh_rate_of_bip - model_share)
    target_share = min(max(target_share, 0.0), 1.0)
    target_p_xbh = target_share * p_bip
    target_p_non_xbh_bip = p_bip - target_p_xbh

    xbh_scale = (target_p_xbh / p_xbh) if p_xbh > 1e-8 else 1.0
    non_xbh_scale = (target_p_non_xbh_bip / p_non_xbh_bip) if p_non_xbh_bip > 1e-8 else 1.0

    result = dict(distribution)
    for o in XBH_OUTCOMES:
        if o in result:
            result[o] = result[o] * xbh_scale
    for o in NON_XBH_BIP_OUTCOMES:
        if o in result:
            result[o] = result[o] * non_xbh_scale
    return result


def apply_xbh_calibration_correction_batched(probs: torch.Tensor, real_xbh_rate_of_bip: torch.Tensor, gain: float) -> torch.Tensor:
    """Vectorized counterpart to apply_xbh_calibration_correction --
    `probs`: [N, len(OUTCOME_VOCAB)] (same shape
    batched_event_outcome_distribution returns). `real_xbh_rate_of_bip`:
    [N], one real leak-safe rate per row (varies with each row's own active
    pitcher). Pure multiplicative rescaling within each of the two
    sub-groups (XBH vs. non-XBH-BIP), which conserves p_bip and total
    probability exactly by construction -- see module docstring."""
    if gain == 0.0:
        return probs
    device = probs.device
    bip_idx = torch.tensor(BIP_INDICES, device=device)
    xbh_idx = torch.tensor(XBH_INDICES, device=device)
    non_xbh_bip_idx = torch.tensor(NON_XBH_BIP_INDICES, device=device)

    p_bip = probs.index_select(1, bip_idx).sum(dim=1)
    p_xbh = probs.index_select(1, xbh_idx).sum(dim=1)
    p_non_xbh_bip = p_bip - p_xbh

    safe_p_bip = p_bip.clamp(min=1e-8)
    model_share = p_xbh / safe_p_bip
    target_share = (model_share + gain * (real_xbh_rate_of_bip - model_share)).clamp(0.0, 1.0)
    target_p_xbh = target_share * p_bip
    target_p_non_xbh_bip = p_bip - target_p_xbh

    xbh_scale = torch.where(p_xbh > 1e-8, target_p_xbh / p_xbh.clamp(min=1e-8), torch.ones_like(p_xbh))
    non_xbh_scale = torch.where(p_non_xbh_bip > 1e-8, target_p_non_xbh_bip / p_non_xbh_bip.clamp(min=1e-8), torch.ones_like(p_non_xbh_bip))
    # No BIP probability at all (p_bip ~ 0) -- nothing to correct, leave that row untouched.
    xbh_scale = torch.where(p_bip > 1e-8, xbh_scale, torch.ones_like(p_bip))
    non_xbh_scale = torch.where(p_bip > 1e-8, non_xbh_scale, torch.ones_like(p_bip))

    result = probs.clone()
    result[:, xbh_idx] = probs.index_select(1, xbh_idx) * xbh_scale.unsqueeze(1)
    result[:, non_xbh_bip_idx] = probs.index_select(1, non_xbh_bip_idx) * non_xbh_scale.unsqueeze(1)
    return result
