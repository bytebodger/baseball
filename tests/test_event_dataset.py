import numpy as np
import pandas as pd
import pytest
import torch

from src.data.contact_quality import ContactQualityHistory, build_contact_quality_history
from src.data.event_dataset import (
    BASE_STATE_COLUMNS,
    CONTACT_QUALITY_FEATURE_NAMES,
    CONTEXT_DIM,
    SITUATIONAL_CONTINUOUS_FEATURES,
    EventBatchCollator,
    EventDataset,
    compute_contact_quality_stats,
    compute_situational_stats,
)
from src.data.event_embedding_cache import EmbeddingCache, precompute_and_cache_embeddings
from src.data.park_factors import ParkFactorConfig, ParkFactorEmbedding, compute_league_rates, compute_park_factors
from src.data.sequence_dataset import MATCHUP_INDEX, OUTCOME_INDEX
from src.models.long_history_encoder import CareerEncoder, CareerEncoderConfig, ChunkEncoder, ChunkEncoderConfig, LongHistoryEncoder


def _pitches_fixture() -> pd.DataFrame:
    """Two seasons (2022, 2023) at one park, enough games for park_factors'
    rolling window to produce a real (non-fallback) factor for 2023."""
    rows = []

    def add(pitcher, batter, season, date, at_bat, pitch_num, outs, balls, strikes, on1, on2, on3,
            home_score, away_score, tto, inning, stand, p_throws, outcome, inning_topbot="Top"):
        rows.append(
            {
                "pitcher_id": pitcher,
                "batter_id": batter,
                "game_date": pd.Timestamp(date),
                "game_pk": hash((season, date, at_bat)) % 100000,
                "at_bat_number": at_bat,
                "pitch_number": pitch_num,
                "pitch_type": "FF",
                "release_speed": 92.0,
                "spin_rate": 2200.0,
                "plate_x": 0.1,
                "plate_z": 2.3,
                "outs_when_up": outs,
                "balls": balls,
                "strikes": strikes,
                "on_1b": on1,
                "on_2b": on2,
                "on_3b": on3,
                "home_score": home_score,
                "away_score": away_score,
                "times_through_order": tto,
                "inning": inning,
                "inning_topbot": inning_topbot,
                "stand": stand,
                "p_throws": p_throws,
                "park_id": "PARK_A",
                "season": season,
                "outcome": outcome,
            }
        )

    # 2022: a handful of games so 2023's rolling park factor has a real prior season.
    for i in range(5):
        add(1, 10, 2022, f"2022-04-{i+1:02d}", 1, 1, 0, 0, 0, None, None, None, 0, 0, 0, 1, "R", "R", "home_run")
    # 2023: the season under test. The 04-12 row is "Top" (away team
    # batting) with home ahead 4-1, so its batting-team-relative score_diff
    # (away_score - home_score = -3) differs in sign from home-minus-away
    # (+3) -- a real regression check against reverting to the old formula.
    add(1, 10, 2023, "2023-04-05", 1, 1, 0, 0, 0, None, None, None, 0, 0, 0, 1, "R", "R", "ball")
    add(1, 10, 2023, "2023-04-05", 1, 2, 1, 1, 1, 555, None, None, 2, 1, 0, 3, "R", "R", "called_strike")
    add(1, 11, 2023, "2023-04-12", 2, 1, 2, 3, 2, 555, 556, 557, 4, 1, 1, 7, "L", "R", "strikeout", inning_topbot="Top")
    add(2, 10, 2023, "2023-04-20", 1, 1, 0, 0, 0, None, None, None, 0, 0, 0, 1, "R", "L", "single")

    return pd.DataFrame(rows)


def _small_encoder_and_configs():
    chunk_config = ChunkEncoderConfig(hidden_size=8, num_layers=1, num_heads=2, dropout=0.0, feedforward_dim=16, max_seq_len=10)
    career_config = CareerEncoderConfig(hidden_size=8, num_layers=1, num_heads=2, dropout=0.0, feedforward_dim=16, max_chunks=6)
    torch.manual_seed(0)
    encoder = LongHistoryEncoder(ChunkEncoder(chunk_config), CareerEncoder(career_config))
    return encoder, chunk_config, career_config


def _build_park_and_league(pitches):
    config = ParkFactorConfig(rolling_years=3, embedding_dim=4)
    park_factors = compute_park_factors(pitches, rolling_years=config.rolling_years)
    park_factor_embedding = ParkFactorEmbedding(config, park_factors)
    league_rates = compute_league_rates(pitches, rolling_years=config.rolling_years)
    return park_factor_embedding, league_rates


def _empty_contact_quality() -> ContactQualityHistory:
    """No per-player history at all -- every lookup falls back to the fixed
    league average, giving a deterministic constant contact-quality value
    across every row. Fine for tests that aren't exercising contact-quality
    behavior specifically (see test_event_dataset_contact_quality_* below
    for tests that build a real per-pitcher history instead)."""
    return ContactQualityHistory(
        {}, {}, {}, league_avg_exit_velo=90.0, league_avg_hard_hit_rate=0.3,
        babip_dates_by_player={}, babip_hit_by_player={}, league_avg_babip=0.3,
    )


def _dummy_contact_quality_stats() -> dict[str, tuple[float, float]]:
    return {"pitcher_exit_velo": (90.0, 1.0), "batter_exit_velo": (90.0, 1.0)}


def _build_dataset(pitches, stats, park_factor_embedding, league_rates, pitcher_cq=None, batter_cq=None, cq_stats=None, min_events=None):
    return EventDataset(
        pitches, stats, park_factor_embedding, league_rates,
        pitcher_cq or _empty_contact_quality(), batter_cq or _empty_contact_quality(), cq_stats or _dummy_contact_quality_stats(),
        contact_quality_min_events=0 if min_events is None else min_events,
    )


# ---------- compute_situational_stats ----------


def test_compute_situational_stats_covers_every_declared_feature():
    pitches = _pitches_fixture()
    stats = compute_situational_stats(pitches)
    assert set(stats.keys()) == set(SITUATIONAL_CONTINUOUS_FEATURES)
    for mean, std in stats.values():
        assert std > 0


def test_compute_situational_stats_score_diff_is_batting_team_relative():
    pitches = _pitches_fixture()
    stats = compute_situational_stats(pitches)
    is_away_batting = pitches["inning_topbot"] == "Top"
    batting_score = pitches["away_score"].where(is_away_batting, pitches["home_score"])
    fielding_score = pitches["home_score"].where(is_away_batting, pitches["away_score"])
    expected_mean = (batting_score - fielding_score).astype("float64").mean()
    assert stats["score_diff"][0] == pytest.approx(expected_mean)


def test_score_diff_sign_flips_between_batting_team_relative_and_home_minus_away():
    """Regression check: the 2023-04-12 row is "Top" (away batting) with
    home ahead 4-1 -- home-minus-away would be +3, but the actual batting
    team (away) is down 3, so the correct, current score_diff is -3."""
    pitches = _pitches_fixture()
    stats = {col: (0.0, 1.0) for col in SITUATIONAL_CONTINUOUS_FEATURES}  # no normalization, read raw value
    park_factor_embedding, league_rates = _build_park_and_league(pitches)
    dataset = _build_dataset(pitches, stats, park_factor_embedding, league_rates)

    idx = pitches.index[pitches["game_date"] == pd.Timestamp("2023-04-12")][0]
    score_diff_col = SITUATIONAL_CONTINUOUS_FEATURES.index("score_diff")
    assert dataset.situational[idx, score_diff_col].item() == pytest.approx(-3.0)
    assert (pitches.loc[idx, "home_score"] - pitches.loc[idx, "away_score"]) == 3  # old formula would've said +3


# ---------- EventDataset ----------


def test_event_dataset_base_state_flags_reflect_runner_presence():
    pitches = _pitches_fixture()
    stats = compute_situational_stats(pitches)
    park_factor_embedding, league_rates = _build_park_and_league(pitches)
    dataset = _build_dataset(pitches, stats, park_factor_embedding, league_rates)

    # Row 5 (2023-04-12 at_bat 2): runners on 1st, 2nd, and 3rd.
    idx = pitches.index[(pitches["game_date"] == pd.Timestamp("2023-04-12"))][0]
    assert torch.equal(dataset.base_state[idx], torch.tensor([1.0, 1.0, 1.0]))

    # Row 1 (first 2023-04-05 pitch): bases empty.
    idx_empty = pitches.index[(pitches["game_date"] == pd.Timestamp("2023-04-05")) & (pitches["pitch_number"] == 1)][0]
    assert torch.equal(dataset.base_state[idx_empty], torch.tensor([0.0, 0.0, 0.0]))


def test_event_dataset_matchup_index_matches_stand_p_throws():
    pitches = _pitches_fixture()
    stats = compute_situational_stats(pitches)
    park_factor_embedding, league_rates = _build_park_and_league(pitches)
    dataset = _build_dataset(pitches, stats, park_factor_embedding, league_rates)

    idx = pitches.index[pitches["game_date"] == pd.Timestamp("2023-04-12")][0]
    assert dataset.matchup_index[idx].item() == MATCHUP_INDEX["L_R"]  # stand=L, p_throws=R


def test_event_dataset_target_matches_outcome_index():
    pitches = _pitches_fixture()
    stats = compute_situational_stats(pitches)
    park_factor_embedding, league_rates = _build_park_and_league(pitches)
    dataset = _build_dataset(pitches, stats, park_factor_embedding, league_rates)

    idx = pitches.index[pitches["outcome"] == "strikeout"][0]
    assert dataset.target[idx].item() == OUTCOME_INDEX["strikeout"]


def test_event_dataset_situational_features_are_z_scored_with_zero_mean_on_train_stats():
    pitches = _pitches_fixture()
    stats = compute_situational_stats(pitches)
    park_factor_embedding, league_rates = _build_park_and_league(pitches)
    dataset = _build_dataset(pitches, stats, park_factor_embedding, league_rates)

    # Using the same pitches to compute stats and features means the
    # per-column mean of the z-scored tensor should be ~0.
    assert dataset.situational.mean(dim=0).abs().max().item() < 1e-5


def test_event_dataset_getitem_shapes():
    pitches = _pitches_fixture()
    stats = compute_situational_stats(pitches)
    park_factor_embedding, league_rates = _build_park_and_league(pitches)
    dataset = _build_dataset(pitches, stats, park_factor_embedding, league_rates)

    sample = dataset[0]
    assert sample["situational"].shape == (len(SITUATIONAL_CONTINUOUS_FEATURES),)
    assert sample["base_state"].shape == (len(BASE_STATE_COLUMNS),)
    assert sample["league_rates"].shape == (2,)
    assert sample["contact_quality"].shape == (len(CONTACT_QUALITY_FEATURE_NAMES),)
    assert sample["contact_quality_aux_target"].shape == (2,)
    assert sample["matchup_index"].dim() == 0
    assert sample["park_index"].dim() == 0
    assert sample["target"].dim() == 0


# ---------- contact-quality wiring ----------


def _batted_ball_fixture() -> pd.DataFrame:
    """pitcher 1 allows consistently weak contact (low exit velo, never
    hard-hit, all singles); pitcher 2 allows consistently hard contact (all
    home runs) -- a real differentiation the wired-in feature should
    reflect, matched against _pitches_fixture's pitcher_id=1/2 rows."""
    rows = []
    for i in range(3):
        rows.append({
            "pitcher_id": 1, "batter_id": 10, "game_date": pd.Timestamp(f"2023-01-{i+1:02d}"),
            "launch_speed": 80.0, "hard_hit": 0.0, "outcome": "single", "is_home_run": False, "is_babip_hit": 1.0,
        })
    for i in range(3):
        rows.append({
            "pitcher_id": 2, "batter_id": 10, "game_date": pd.Timestamp(f"2023-01-{i+1:02d}"),
            "launch_speed": 105.0, "hard_hit": 1.0, "outcome": "home_run", "is_home_run": True, "is_babip_hit": 0.0,
        })
    return pd.DataFrame(rows)


def test_compute_contact_quality_stats_returns_the_two_declared_keys():
    pitches = _pitches_fixture()
    pitcher_cq = build_contact_quality_history(_batted_ball_fixture(), "pitcher_id")
    batter_cq = build_contact_quality_history(_batted_ball_fixture(), "batter_id")
    stats = compute_contact_quality_stats(pitches, pitcher_cq, batter_cq, min_events=0)
    assert set(stats.keys()) == {"pitcher_exit_velo", "batter_exit_velo"}
    for mean, std in stats.values():
        assert std > 0


def test_event_dataset_contact_quality_differentiates_pitchers_with_real_history():
    pitches = _pitches_fixture()
    stats = compute_situational_stats(pitches)
    park_factor_embedding, league_rates = _build_park_and_league(pitches)
    pitcher_cq = build_contact_quality_history(_batted_ball_fixture(), "pitcher_id")
    batter_cq = build_contact_quality_history(_batted_ball_fixture(), "batter_id")
    cq_stats = compute_contact_quality_stats(pitches, pitcher_cq, batter_cq, min_events=0)
    dataset = _build_dataset(pitches, stats, park_factor_embedding, league_rates, pitcher_cq, batter_cq, cq_stats, min_events=0)

    # Every 2023 row's cutoff (April+) is well after pitcher 1/2's January
    # batted-ball history, so both get a real (non-fallback) value -- and
    # pitcher 2's (hard contact allowed) exit-velo-allowed feature must be
    # higher than pitcher 1's (weak contact allowed).
    pitcher1_idx = pitches.index[pitches["pitcher_id"] == 1][0]
    pitcher2_idx = pitches.index[pitches["pitcher_id"] == 2][0]
    assert dataset.contact_quality[pitcher2_idx, 0].item() > dataset.contact_quality[pitcher1_idx, 0].item()
    assert dataset.contact_quality[pitcher2_idx, 1].item() > dataset.contact_quality[pitcher1_idx, 1].item()  # hard-hit rate


def test_event_dataset_contact_quality_aux_target_reflects_real_babip_and_hard_hit_rate():
    pitches = _pitches_fixture()
    stats = compute_situational_stats(pitches)
    park_factor_embedding, league_rates = _build_park_and_league(pitches)
    pitcher_cq = build_contact_quality_history(_batted_ball_fixture(), "pitcher_id")
    batter_cq = build_contact_quality_history(_batted_ball_fixture(), "batter_id")
    cq_stats = compute_contact_quality_stats(pitches, pitcher_cq, batter_cq, min_events=0)
    dataset = _build_dataset(pitches, stats, park_factor_embedding, league_rates, pitcher_cq, batter_cq, cq_stats, min_events=0)

    # Pitcher 1's 3 batted balls (_batted_ball_fixture) are all singles --
    # BABIP=1.0. Pitcher 2's 3 are all home runs, excluded entirely from
    # BABIP, so pitcher 2 has zero babip-relevant history and falls back to
    # the league average.
    pitcher1_idx = pitches.index[pitches["pitcher_id"] == 1][0]
    pitcher2_idx = pitches.index[pitches["pitcher_id"] == 2][0]
    assert dataset.contact_quality_aux_target[pitcher1_idx, 0].item() == pytest.approx(1.0)
    assert dataset.contact_quality_aux_target[pitcher2_idx, 0].item() == pytest.approx(pitcher_cq.league_avg_babip)

    # The hard-hit-rate column of the aux target must match the (un-
    # z-scored) hard-hit-rate INPUT feature exactly -- same underlying
    # rolling stat, looked up the same way.
    assert dataset.contact_quality_aux_target[pitcher1_idx, 1].item() == pytest.approx(dataset.contact_quality[pitcher1_idx, 1].item())
    assert dataset.contact_quality_aux_target[pitcher2_idx, 1].item() == pytest.approx(dataset.contact_quality[pitcher2_idx, 1].item())


def test_event_dataset_contact_quality_exit_velo_is_z_scored_with_zero_mean_on_train_stats():
    pitches = _pitches_fixture()
    stats = compute_situational_stats(pitches)
    park_factor_embedding, league_rates = _build_park_and_league(pitches)
    pitcher_cq = build_contact_quality_history(_batted_ball_fixture(), "pitcher_id")
    batter_cq = build_contact_quality_history(_batted_ball_fixture(), "batter_id")
    cq_stats = compute_contact_quality_stats(pitches, pitcher_cq, batter_cq, min_events=0)
    dataset = _build_dataset(pitches, stats, park_factor_embedding, league_rates, pitcher_cq, batter_cq, cq_stats, min_events=0)

    pitcher_exit_velo_col = 0
    assert dataset.contact_quality[:, pitcher_exit_velo_col].mean().abs().item() < 1e-5


# ---------- EventBatchCollator ----------


def test_event_batch_collator_produces_correctly_shaped_batch(tmp_path):
    pitches = _pitches_fixture()
    stats = compute_situational_stats(pitches)
    park_factor_embedding, league_rates = _build_park_and_league(pitches)
    dataset = _build_dataset(pitches, stats, park_factor_embedding, league_rates)

    encoder, chunk_config, career_config = _small_encoder_and_configs()
    precompute_and_cache_embeddings(
        pitches, encoder, tmp_path, career_config.max_chunks, chunk_config.max_seq_len, device=torch.device("cpu"), batch_size=4
    )
    pitcher_cache = EmbeddingCache(tmp_path, "pitcher")
    batter_cache = EmbeddingCache(tmp_path, "batter")
    collate_fn = EventBatchCollator(pitcher_cache, batter_cache)

    batch = collate_fn([dataset[i] for i in range(len(dataset))])
    n = len(dataset)
    assert batch["pitcher_embedding"].shape == (n, career_config.hidden_size)
    assert batch["batter_embedding"].shape == (n, career_config.hidden_size)
    assert batch["context"].shape == (n, CONTEXT_DIM)
    assert batch["contact_quality_aux_target"].shape == (n, 2)
    assert batch["matchup_index"].shape == (n,)
    assert batch["park_index"].shape == (n,)
    assert batch["target"].shape == (n,)


def test_event_batch_collator_raises_on_a_pair_missing_from_the_cache(tmp_path):
    pitches = _pitches_fixture()
    stats = compute_situational_stats(pitches)
    park_factor_embedding, league_rates = _build_park_and_league(pitches)
    dataset = _build_dataset(pitches, stats, park_factor_embedding, league_rates)

    # Deliberately never populate the cache -- every lookup should miss.
    pitcher_cache = EmbeddingCache(tmp_path, "pitcher")
    batter_cache = EmbeddingCache(tmp_path, "batter")
    collate_fn = EventBatchCollator(pitcher_cache, batter_cache)

    with pytest.raises(KeyError):
        collate_fn([dataset[0]])
