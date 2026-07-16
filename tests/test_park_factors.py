import math
import random

import pandas as pd
import pytest
import torch

from src.data import statcast_common as sc
from src.data.park_factors import (
    UNKNOWN_PARK_INDEX,
    LeagueRatesIndex,
    ParkFactorConfig,
    ParkFactorEmbedding,
    compute_game_totals,
    compute_league_rates,
    compute_league_rolling_rates,
    compute_park_factors,
    compute_rolling_park_factors,
    compute_season_park_totals,
    league_rates_for,
)


# ---------- compute_game_totals ----------


def _pitches_for_games():
    # Game 1 at PARK_A/2022: final score 3-2 (total_runs=5), one home_run.
    # Game 2 at PARK_B/2022: final score 4-4 (total_runs=8), two home_runs.
    return pd.DataFrame(
        {
            "game_pk": [1, 1, 1, 2, 2, 2],
            "park_id": ["PARK_A", "PARK_A", "PARK_A", "PARK_B", "PARK_B", "PARK_B"],
            "season": [2022, 2022, 2022, 2022, 2022, 2022],
            "home_score": [0, 1, 3, 0, 2, 4],
            "away_score": [0, 0, 2, 0, 1, 4],
            "outcome": ["home_run", "strikeout", "walk", "home_run", "home_run", "field_out"],
        }
    )


def test_compute_game_totals_uses_max_score_and_counts_home_runs():
    result = compute_game_totals(_pitches_for_games()).set_index("game_pk")
    assert result.loc[1, "park_id"] == "PARK_A"
    assert result.loc[1, "total_runs"] == 5
    assert result.loc[1, "home_runs"] == 1
    assert result.loc[2, "total_runs"] == 8
    assert result.loc[2, "home_runs"] == 2


# ---------- compute_season_park_totals ----------


def test_compute_season_park_totals_aggregates_across_games_at_the_same_park():
    game_totals = pd.DataFrame(
        {
            "game_pk": [1, 2, 3],
            "park_id": ["PARK_A", "PARK_A", "PARK_B"],
            "season": [2022, 2022, 2022],
            "total_runs": [5, 7, 8],
            "home_runs": [1, 3, 2],
        }
    )
    result = compute_season_park_totals(game_totals).set_index(["park_id", "season"])
    assert result.loc[("PARK_A", 2022), "n_games"] == 2
    assert result.loc[("PARK_A", 2022), "total_runs"] == 12
    assert result.loc[("PARK_A", 2022), "total_home_runs"] == 4
    assert result.loc[("PARK_B", 2022), "n_games"] == 1


# ---------- compute_rolling_park_factors ----------


def _season_totals(rows: list[tuple[str, int, int, int, int]]) -> pd.DataFrame:
    """rows: (park_id, season, n_games, total_runs, total_home_runs)"""
    return pd.DataFrame(rows, columns=["park_id", "season", "n_games", "total_runs", "total_home_runs"])


def test_hr_factor_above_one_for_a_park_that_out_homers_the_league():
    # Both parks have an identical prior (2021) season so 2022 has a valid
    # strictly-prior window; PARK_A hits 2 HR/game in 2021, PARK_B hits 0.
    # League average in the 2021-only window is 1 HR/game; PARK_A's 2022
    # factor should be 2.0, PARK_B's 0.0 -- entirely from 2021, none of it
    # from either park's own 2022 games (see the leakage tests below).
    totals = _season_totals(
        [
            ("PARK_A", 2021, 10, 50, 20),
            ("PARK_B", 2021, 10, 50, 0),
            ("PARK_A", 2022, 10, 999, 999),
            ("PARK_B", 2022, 10, 999, 999),
        ]
    )
    result = compute_rolling_park_factors(totals, rolling_years=3)
    result_2022 = result[result["season"] == 2022].set_index("park_id")
    assert result_2022.loc["PARK_A", "hr_rate"] == pytest.approx(2.0)
    assert result_2022.loc["PARK_A", "league_hr_rate"] == pytest.approx(1.0)
    assert result_2022.loc["PARK_A", "hr_factor"] == pytest.approx(2.0)
    assert result_2022.loc["PARK_B", "hr_factor"] == pytest.approx(0.0)


def test_runs_factor_relative_to_league_average():
    totals = _season_totals(
        [
            ("PARK_A", 2021, 10, 100, 10),  # 10 runs/game
            ("PARK_B", 2021, 10, 60, 10),  # 6 runs/game
            ("PARK_A", 2022, 10, 999, 999),
            ("PARK_B", 2022, 10, 999, 999),
        ]
    )
    result = compute_rolling_park_factors(totals, rolling_years=3)
    result_2022 = result[result["season"] == 2022].set_index("park_id")
    # league runs/game (2021 only) = 160/20 = 8
    assert result_2022.loc["PARK_A", "runs_factor"] == pytest.approx(10 / 8)
    assert result_2022.loc["PARK_B", "runs_factor"] == pytest.approx(6 / 8)


def test_rolling_window_sums_trailing_seasons_strictly_before_the_target_season():
    totals = _season_totals(
        [
            ("PARK_A", 2019, 10, 40, 10),
            ("PARK_A", 2020, 10, 40, 10),
            ("PARK_A", 2021, 10, 40, 10),
            ("PARK_A", 2022, 10, 40, 10),
            ("PARK_A", 2023, 10, 40, 10),  # 5th season -- 2019 should be out of range either way
        ]
    )
    result = compute_rolling_park_factors(totals, rolling_years=3).set_index("season")
    # 2022's window covers 2019-2021 (3 seasons strictly before 2022): 30 games, 30 HR -> rate 1.0
    assert result.loc[2022, "rolling_games"] == 30
    assert result.loc[2022, "hr_rate"] == pytest.approx(1.0)
    # 2023's window covers 2020-2022 (2019 rolled off, 2023 itself excluded): still 30 games, 30 HR
    assert result.loc[2023, "rolling_games"] == 30


def test_expanding_window_for_a_park_with_only_one_prior_season():
    # PARK_NEW debuted in 2022 (one season of its own history) and is being
    # looked up for 2023; rolling_years=3 should use just that one prior
    # season (min_periods=1 on the strictly-prior window), not NaN and not
    # 2023's own games.
    totals = _season_totals([("PARK_NEW", 2022, 10, 40, 10), ("PARK_NEW", 2023, 500, 99999, 99999)])
    result = compute_rolling_park_factors(totals, rolling_years=3)
    row = result[result["season"] == 2023].iloc[0]
    assert row["rolling_games"] == 10
    assert row["hr_rate"] == pytest.approx(1.0)
    assert not math.isnan(row["hr_factor"])


def test_a_parks_debut_season_is_dropped_entirely_no_prior_season_to_roll_over():
    # PARK_NEW's *first* tracked season has zero strictly-prior seasons of
    # its own -- nothing legitimate to compute a pre-game factor from -- so
    # it must not appear in the output at all (as opposed to a leaked
    # self-referential "expanding window" row using its own season's games).
    totals = _season_totals([("PARK_NEW", 2023, 10, 40, 10)])
    result = compute_rolling_park_factors(totals, rolling_years=3)
    assert result.empty


# ---------- explicit no-leakage checks ----------
#
# Mirrors how leakage was tested for PlayerPitchSequenceDataset/GameOutcomeDataset
# elsewhere in this project: prove a season's own games play no part in its
# own factor by holding every strictly-prior season fixed and swapping that
# season's own totals for something wildly different, then asserting the
# factor doesn't move at all.


def test_park_factor_for_season_s_is_unaffected_by_season_ss_own_totals():
    prior_seasons = [
        ("PARK_A", 2020, 10, 40, 10),
        ("PARK_A", 2021, 10, 40, 10),
        ("PARK_B", 2020, 10, 40, 10),
        ("PARK_B", 2021, 10, 40, 10),
    ]
    totals_modest_2022 = _season_totals(prior_seasons + [("PARK_A", 2022, 10, 40, 10), ("PARK_B", 2022, 10, 40, 10)])
    totals_outlier_2022 = _season_totals(
        prior_seasons + [("PARK_A", 2022, 500, 999999, 999999), ("PARK_B", 2022, 10, 40, 10)]
    )

    # Sanity: the two inputs really do disagree about 2022's own totals for PARK_A.
    assert totals_modest_2022.set_index(["park_id", "season"]).loc[("PARK_A", 2022), "total_home_runs"] != (
        totals_outlier_2022.set_index(["park_id", "season"]).loc[("PARK_A", 2022), "total_home_runs"]
    )

    result_modest = compute_rolling_park_factors(totals_modest_2022, rolling_years=3)
    result_outlier = compute_rolling_park_factors(totals_outlier_2022, rolling_years=3)

    row_modest = result_modest[(result_modest["park_id"] == "PARK_A") & (result_modest["season"] == 2022)].iloc[0]
    row_outlier = result_outlier[(result_outlier["park_id"] == "PARK_A") & (result_outlier["season"] == 2022)].iloc[0]

    assert row_modest["rolling_games"] == row_outlier["rolling_games"]
    assert row_modest["hr_rate"] == pytest.approx(row_outlier["hr_rate"])
    assert row_modest["hr_factor"] == pytest.approx(row_outlier["hr_factor"])
    # PARK_A's 2022 outlier also would have inflated the leaguewide 2022
    # total if leaked -- confirm the league denominator didn't move either.
    assert row_modest["league_hr_rate"] == pytest.approx(row_outlier["league_hr_rate"])


def test_league_rolling_rate_for_season_s_is_unaffected_by_season_ss_own_totals():
    prior_seasons = [("PARK_A", 2020, 10, 40, 10), ("PARK_A", 2021, 10, 40, 10)]
    totals_modest = _season_totals(prior_seasons + [("PARK_A", 2022, 10, 40, 10)])
    totals_outlier = _season_totals(prior_seasons + [("PARK_A", 2022, 500, 999999, 999999)])

    league_modest = compute_league_rolling_rates(totals_modest, rolling_years=3).set_index("season")
    league_outlier = compute_league_rolling_rates(totals_outlier, rolling_years=3).set_index("season")

    assert league_modest.loc[2022, "league_hr_rate"] == pytest.approx(league_outlier.loc[2022, "league_hr_rate"])
    assert league_modest.loc[2022, "league_runs_rate"] == pytest.approx(league_outlier.loc[2022, "league_runs_rate"])


def test_park_factor_embedding_index_for_never_resolves_to_a_row_derived_from_the_queried_seasons_own_games():
    """End-to-end version of the leakage check through the same surface
    Phase 4 will actually call: build the real ParkFactorEmbedding from two
    inputs that only disagree on season 2022's own totals, and confirm the
    embedding vector looked up for ("PARK_A", 2022) is identical either way.
    """
    prior_seasons = [("PARK_A", 2020, 10, 40, 10), ("PARK_A", 2021, 10, 40, 10)]
    totals_modest = _season_totals(prior_seasons + [("PARK_A", 2022, 10, 40, 10)])
    totals_outlier = _season_totals(prior_seasons + [("PARK_A", 2022, 500, 999999, 999999)])

    factors_modest = compute_rolling_park_factors(totals_modest, rolling_years=3)
    factors_outlier = compute_rolling_park_factors(totals_outlier, rolling_years=3)

    # Pin the RNG before each construction so the embedding's unseeded "free"
    # dims (beyond the two seeded from hr_factor/runs_factor) are identical
    # across both instances too -- otherwise they'd differ from independent
    # random init regardless of any leak, defeating the comparison below.
    config = ParkFactorConfig(embedding_dim=4)
    torch.manual_seed(0)
    embedding_modest = ParkFactorEmbedding(config, factors_modest)
    torch.manual_seed(0)
    embedding_outlier = ParkFactorEmbedding(config, factors_outlier)

    index_modest = embedding_modest.index_for("PARK_A", 2022)
    index_outlier = embedding_outlier.index_for("PARK_A", 2022)
    vector_modest = embedding_modest.embedding.weight.data[index_modest]
    vector_outlier = embedding_outlier.embedding.weight.data[index_outlier]
    assert torch.allclose(vector_modest, vector_outlier)


def test_league_rates_for_never_resolves_to_a_rate_derived_from_the_queried_seasons_own_games():
    prior_seasons = [("PARK_A", 2020, 10, 40, 10), ("PARK_A", 2021, 10, 40, 10)]
    totals_modest = _season_totals(prior_seasons + [("PARK_A", 2022, 10, 40, 10)])
    totals_outlier = _season_totals(prior_seasons + [("PARK_A", 2022, 500, 999999, 999999)])

    league_modest = compute_league_rolling_rates(totals_modest, rolling_years=3)
    league_outlier = compute_league_rolling_rates(totals_outlier, rolling_years=3)

    result_modest = league_rates_for(pd.Series([2022]), league_modest)
    result_outlier = league_rates_for(pd.Series([2022]), league_outlier)
    assert result_modest["league_hr_rate"].iloc[0] == pytest.approx(result_outlier["league_hr_rate"].iloc[0])


# ---------- compute_league_rolling_rates / league_rates_for ----------


def test_compute_league_rolling_rates_matches_compute_rolling_park_factors_denominator():
    # league_hr_rate/league_runs_rate returned standalone should be exactly
    # the same numbers compute_rolling_park_factors merges onto every park's
    # row -- it's the same computation, just exposed directly. Needs a prior
    # (2021) season for 2022 to survive the leakage-safe dropna at all.
    totals = _season_totals(
        [
            ("PARK_A", 2021, 10, 40, 10),
            ("PARK_B", 2021, 10, 40, 10),
            ("PARK_A", 2022, 10, 100, 10),
            ("PARK_B", 2022, 10, 60, 10),
        ]
    )
    league = compute_league_rolling_rates(totals, rolling_years=3).set_index("season")
    park_factors_2022 = compute_rolling_park_factors(totals, rolling_years=3)
    park_factors_2022 = park_factors_2022[park_factors_2022["season"] == 2022]
    assert league.loc[2022, "league_hr_rate"] == pytest.approx(park_factors_2022["league_hr_rate"].iloc[0])
    assert league.loc[2022, "league_runs_rate"] == pytest.approx(park_factors_2022["league_runs_rate"].iloc[0])


def test_compute_league_rolling_rates_rolls_over_trailing_seasons_strictly_before_the_target_season():
    totals = _season_totals(
        [
            ("PARK_A", 2020, 10, 40, 10),
            ("PARK_B", 2020, 10, 40, 10),
            ("PARK_A", 2021, 10, 40, 10),
            ("PARK_B", 2021, 10, 40, 10),
            # 2022's own (much lower) rate must not leak into 2022's own row.
            ("PARK_A", 2022, 10, 20, 5),
            ("PARK_B", 2022, 10, 20, 5),
        ]
    )
    league = compute_league_rolling_rates(totals, rolling_years=3).set_index("season")
    # 2020 has no prior season of its own at all -> dropped entirely.
    assert 2020 not in league.index
    # 2021's window is 2020 only (its sole strictly-prior season): 20 games, 20 HR -> rate 1.0
    assert league.loc[2021, "league_hr_rate"] == pytest.approx(1.0)
    # 2022's window is 2020-2021 (2022's own lower-rate games excluded): 40 games, 40 HR -> still 1.0,
    # not diluted toward 2022's own 0.5 rate the way it would be if 2022 leaked into its own window.
    assert league.loc[2022, "league_hr_rate"] == pytest.approx(1.0)


def test_league_rates_for_exact_match():
    league_rates = pd.DataFrame({"season": [2021, 2022], "league_hr_rate": [1.0, 0.8], "league_runs_rate": [8.0, 7.5]})
    result = league_rates_for(pd.Series([2021, 2022]), league_rates)
    assert result["league_hr_rate"].tolist() == pytest.approx([1.0, 0.8])
    assert result["league_runs_rate"].tolist() == pytest.approx([8.0, 7.5])


def test_league_rates_for_falls_back_to_latest_known_season():
    league_rates = pd.DataFrame({"season": [2021, 2022], "league_hr_rate": [1.0, 0.8], "league_runs_rate": [8.0, 7.5]})
    result = league_rates_for(pd.Series([2025]), league_rates)
    assert result["league_hr_rate"].iloc[0] == pytest.approx(0.8)


def test_league_rates_for_falls_back_to_earliest_known_season():
    league_rates = pd.DataFrame({"season": [2021, 2022], "league_hr_rate": [1.0, 0.8], "league_runs_rate": [8.0, 7.5]})
    result = league_rates_for(pd.Series([2015]), league_rates)
    assert result["league_hr_rate"].iloc[0] == pytest.approx(1.0)


def test_league_rates_for_preserves_original_order_and_index_with_duplicates():
    league_rates = pd.DataFrame({"season": [2021, 2022], "league_hr_rate": [1.0, 0.8], "league_runs_rate": [8.0, 7.5]})
    season = pd.Series([2022, 2021, 2022], index=[5, 6, 7])
    result = league_rates_for(season, league_rates)
    assert result.index.tolist() == [5, 6, 7]
    assert result["league_hr_rate"].tolist() == pytest.approx([0.8, 1.0, 0.8])


def test_league_rates_index_exact_match():
    league_rates = pd.DataFrame({"season": [2021, 2022], "league_hr_rate": [1.0, 0.8], "league_runs_rate": [8.0, 7.5]})
    index = LeagueRatesIndex(league_rates)
    assert index.for_season(2021) == pytest.approx((1.0, 8.0))
    assert index.for_season(2022) == pytest.approx((0.8, 7.5))


def test_league_rates_index_falls_back_to_latest_known_season():
    league_rates = pd.DataFrame({"season": [2021, 2022], "league_hr_rate": [1.0, 0.8], "league_runs_rate": [8.0, 7.5]})
    index = LeagueRatesIndex(league_rates)
    assert index.for_season(2025) == pytest.approx((0.8, 7.5))


def test_league_rates_index_falls_back_to_earliest_known_season():
    league_rates = pd.DataFrame({"season": [2021, 2022], "league_hr_rate": [1.0, 0.8], "league_runs_rate": [8.0, 7.5]})
    index = LeagueRatesIndex(league_rates)
    assert index.for_season(2015) == pytest.approx((1.0, 8.0))


def test_league_rates_index_matches_league_rates_for_across_every_season_including_gaps_and_both_fallbacks():
    # Deliberately non-contiguous known seasons (a gap at 2020) so the
    # "nearest known season at or before" fallback is actually exercised,
    # not just the exact-match path.
    league_rates = pd.DataFrame(
        {
            "season": [2018, 2019, 2021, 2023],
            "league_hr_rate": [1.0, 1.1, 0.9, 1.2],
            "league_runs_rate": [8.0, 8.2, 7.9, 8.5],
        }
    )
    index = LeagueRatesIndex(league_rates)
    queried = [2010, 2018, 2019, 2020, 2021, 2022, 2023, 2030]
    oracle = league_rates_for(pd.Series(queried), league_rates)
    for season, oracle_hr, oracle_runs in zip(queried, oracle["league_hr_rate"], oracle["league_runs_rate"]):
        hr_rate, runs_rate = index.for_season(season)
        assert hr_rate == pytest.approx(oracle_hr)
        assert runs_rate == pytest.approx(oracle_runs)


def _realistic_pitch_frame() -> pd.DataFrame:
    # Two seasons so the leak-safe pipeline has a strictly-prior season to
    # roll over for 2024: 2023 has one no-HR game per park (game_pk 98/99),
    # 2024 has one HR-scoring game per park (game_pk 100/101). The 2024 row
    # asserted on below must reflect 2023's rates, not its own -- if it did,
    # hr_rate would come out 1.0 (from 2024's own home_run), not 0.0.
    raw = pd.DataFrame(
        {
            "pitcher": [1, 1, 2, 2, 1, 2],
            "batter": [10, 11, 12, 13, 10, 12],
            "game_date": ["2023-06-01", "2023-06-01", "2023-06-02", "2023-06-02", "2024-06-01", "2024-06-02"],
            "game_pk": [98, 98, 99, 99, 100, 101],
            "at_bat_number": [1, 2, 1, 2, 1, 1],
            "pitch_number": [1, 1, 1, 1, 1, 1],
            "pitch_type": ["FF", "FF", "SL", "SL", "FF", "SL"],
            "release_speed": [95.0, 94.0, 85.0, 86.0, 95.0, 85.0],
            "release_spin_rate": [2200, 2200, 2400, 2400, 2200, 2400],
            "spin_rate_deprecated": [None, None, None, None, None, None],
            "plate_x": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "plate_z": [2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
            "balls": [0, 0, 0, 0, 0, 0],
            "strikes": [0, 0, 0, 0, 0, 0],
            "outs_when_up": [0, 1, 0, 1, 0, 0],
            "on_1b": [None, None, None, None, None, None],
            "on_2b": [None, None, None, None, None, None],
            "on_3b": [None, None, None, None, None, None],
            "home_score": [0, 0, 0, 0, 1, 0],
            "away_score": [0, 0, 0, 0, 0, 0],
            "n_thruorder_pitcher": [1, 1, 1, 1, 1, 1],
            "inning": [1, 1, 1, 1, 1, 1],
            "inning_topbot": ["Top"] * 6,
            "stand": ["R", "L", "R", "L", "R", "R"],
            "p_throws": ["R", "R", "R", "R", "R", "R"],
            "home_team": ["DET", "DET", "CLE", "CLE", "DET", "CLE"],
            "away_team": ["CLE", "CLE", "DET", "DET", "CLE", "DET"],
            "game_year": [2023, 2023, 2023, 2023, 2024, 2024],
            "events": ["field_out", "strikeout", "field_out", "walk", "home_run", "strikeout"],
            "description": ["hit_into_play", "swinging_strike", "hit_into_play", "ball", "hit_into_play", "swinging_strike"],
        }
    )
    return sc.build_pitch_frame_from_raw(raw)


def test_compute_park_factors_end_to_end_from_a_realistic_pitch_frame():
    pitches = _realistic_pitch_frame()
    result = compute_park_factors(pitches, rolling_years=3)
    # 2023 is DET's debut tracked season (no prior season of its own) -> dropped.
    assert not ((result["park_id"] == "DET") & (result["season"] == 2023)).any()
    det_2024 = result[(result["park_id"] == "DET") & (result["season"] == 2024)].iloc[0]
    assert det_2024["rolling_games"] == 1
    # DET's 2024 row must reflect 2023's 0-HR game, not 2024's own home_run.
    assert det_2024["hr_rate"] == pytest.approx(0.0)


def test_compute_league_rates_end_to_end_from_a_realistic_pitch_frame():
    pitches = _realistic_pitch_frame()
    result = compute_league_rates(pitches, rolling_years=3)
    assert 2023 not in result["season"].to_numpy()
    row = result[result["season"] == 2024].iloc[0]
    # 2024's row must reflect 2023's 2 games / 0 combined home runs, not
    # 2024's own game_pk 100's home_run.
    assert row["league_hr_rate"] == pytest.approx(0.0)


# ---------- ParkFactorEmbedding ----------


def _sample_park_factors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "park_id": ["PARK_A", "PARK_A", "PARK_B"],
            "season": [2021, 2022, 2022],
            "rolling_games": [10, 20, 10],
            "hr_rate": [1.0, 1.5, 0.5],
            "runs_rate": [8.0, 9.0, 7.0],
            "league_hr_rate": [1.0, 1.0, 1.0],
            "league_runs_rate": [8.0, 8.0, 8.0],
            "hr_factor": [1.0, 1.5, 0.5],
            "runs_factor": [1.0, 1.125, 0.875],
        }
    )


def test_vocab_has_one_row_per_park_season_pair():
    embedding = ParkFactorEmbedding(ParkFactorConfig(embedding_dim=4), _sample_park_factors())
    assert set(embedding.vocab.keys()) == {("PARK_A", 2021), ("PARK_A", 2022), ("PARK_B", 2022)}
    # +1 for the reserved unknown-park row.
    assert embedding.embedding.num_embeddings == 4


def test_forward_returns_one_vector_per_index_with_configured_dim():
    embedding = ParkFactorEmbedding(ParkFactorConfig(embedding_dim=6), _sample_park_factors())
    indices = torch.tensor([1, 2, 3])
    out = embedding(indices)
    assert out.shape == (3, 6)


def test_embedding_is_learnable_via_backprop():
    embedding = ParkFactorEmbedding(ParkFactorConfig(embedding_dim=4), _sample_park_factors())
    indices = torch.tensor([1, 2])
    out = embedding(indices)
    loss = out.sum()
    loss.backward()
    assert embedding.embedding.weight.grad is not None
    assert embedding.embedding.weight.grad.abs().sum() > 0


def test_embedding_dim_two_seeds_directly_from_log_factors():
    embedding = ParkFactorEmbedding(ParkFactorConfig(embedding_dim=2), _sample_park_factors())
    index = embedding.index_for("PARK_A", 2022)
    vector = embedding.embedding.weight.data[index]
    assert vector[0].item() == pytest.approx(math.log(1.5))
    assert vector[1].item() == pytest.approx(math.log(1.125))


def test_index_for_exact_match():
    embedding = ParkFactorEmbedding(ParkFactorConfig(embedding_dim=4), _sample_park_factors())
    assert embedding.index_for("PARK_A", 2021) == embedding.vocab[("PARK_A", 2021)]
    assert embedding.index_for("PARK_A", 2022) == embedding.vocab[("PARK_A", 2022)]


def test_index_for_falls_back_to_latest_known_season_at_or_before_query():
    embedding = ParkFactorEmbedding(ParkFactorConfig(embedding_dim=4), _sample_park_factors())
    # PARK_A has no row for 2025; should fall back to its latest known season, 2022.
    assert embedding.index_for("PARK_A", 2025) == embedding.vocab[("PARK_A", 2022)]


def test_index_for_falls_back_to_earliest_known_season_when_queried_earlier():
    embedding = ParkFactorEmbedding(ParkFactorConfig(embedding_dim=4), _sample_park_factors())
    # PARK_A's earliest known season is 2021; a 2015 query has nothing earlier to use.
    assert embedding.index_for("PARK_A", 2015) == embedding.vocab[("PARK_A", 2021)]


def test_index_for_unknown_park_returns_unknown_index():
    embedding = ParkFactorEmbedding(ParkFactorConfig(embedding_dim=4), _sample_park_factors())
    assert embedding.index_for("NEVER_SEEN_PARK", 2022) == UNKNOWN_PARK_INDEX


def test_indices_for_vectorized_matches_index_for_row_by_row():
    embedding = ParkFactorEmbedding(ParkFactorConfig(embedding_dim=4), _sample_park_factors())
    park_ids = pd.Series(["PARK_A", "PARK_B", "NEVER_SEEN_PARK", "PARK_A"])
    seasons = pd.Series([2021, 2022, 2022, 2025])
    result = embedding.indices_for(park_ids, seasons)
    expected = [embedding.index_for(p, s) for p, s in zip(park_ids, seasons)]
    assert result.tolist() == expected


def test_indices_for_preserves_original_index_order_with_duplicates():
    embedding = ParkFactorEmbedding(ParkFactorConfig(embedding_dim=4), _sample_park_factors())
    park_ids = pd.Series(["PARK_A", "PARK_A", "PARK_A"])
    seasons = pd.Series([2021, 2021, 2022])
    result = embedding.indices_for(park_ids, seasons)
    assert result.tolist() == [
        embedding.vocab[("PARK_A", 2021)],
        embedding.vocab[("PARK_A", 2021)],
        embedding.vocab[("PARK_A", 2022)],
    ]


def _multi_park_factors() -> pd.DataFrame:
    """5 known parks x 2019-2022, for a random batch with real variety."""
    parks = ["PARK_A", "PARK_B", "PARK_C", "PARK_D", "PARK_E"]
    rows = [
        {
            "park_id": park,
            "season": season,
            "rolling_games": 20,
            "hr_rate": 1.0,
            "runs_rate": 8.0,
            "league_hr_rate": 1.0,
            "league_runs_rate": 8.0,
            "hr_factor": 1.0 + 0.05 * i,
            "runs_factor": 1.0 + 0.02 * i,
        }
        for i, park in enumerate(parks)
        for season in range(2019, 2023)
    ]
    return pd.DataFrame(rows)


def test_indices_for_matches_index_for_on_a_random_batch_of_32_pairs_covering_both_fallbacks():
    embedding = ParkFactorEmbedding(ParkFactorConfig(embedding_dim=4), _multi_park_factors())
    known_parks = ["PARK_A", "PARK_B", "PARK_C", "PARK_D", "PARK_E"]

    rng = random.Random(42)
    park_ids = ["NEVER_SEEN_PARK", "PARK_A"]  # guarantee: unknown-park fallback, latest-known-season fallback
    seasons = [2021, 2030]  # 2030 is well past PARK_A's latest known season (2022)
    for _ in range(30):
        park_ids.append(rng.choice(known_parks + ["NEVER_SEEN_PARK"]))
        seasons.append(rng.choice(range(2015, 2031)))  # spans before/within/after the known 2019-2022 range

    assert len(park_ids) == 32
    park_series = pd.Series(park_ids)
    season_series = pd.Series(seasons)

    batched = embedding.indices_for(park_series, season_series)
    individual = [embedding.index_for(p, s) for p, s in zip(park_ids, seasons)]
    assert batched.tolist() == individual

    # Confirm the batch actually exercised both fallback paths, not just exact matches.
    assert individual[0] == UNKNOWN_PARK_INDEX
    assert individual[1] == embedding.vocab[("PARK_A", 2022)]
    assert any(p == "NEVER_SEEN_PARK" for p in park_ids[2:])
    assert any(s > 2022 or s < 2019 for s in seasons[2:])
