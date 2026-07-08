import numpy as np
import pandas as pd
import pytest

from src.data.build_betting_lines import (
    CROSSWALK_CONFIG_PATH,
    _normalize_team_token,
    american_odds_to_implied_prob,
    build_betting_lines,
    load_team_crosswalk,
    no_vig_home_win_prob,
)

ODDS_DEFAULTS = dict(
    away_ml_open=100, away_ml_close=105, home_ml_open=-110, home_ml_close=-115,
    total_open=8.5, over_open_odds=-105, under_open_odds=-115,
    total_close=8.0, over_close_odds=-110, under_close_odds=-110,
)


def _betting_row(source_game_id, raw_date, away_team, home_team, away_score, home_score, **overrides):
    row = dict(
        source_game_id=source_game_id,
        raw_date=raw_date,
        raw_away_team=away_team,
        raw_home_team=home_team,
        game_date=pd.Timestamp(raw_date),
        away_team=away_team,
        home_team=home_team,
        away_score=away_score,
        home_score=home_score,
        **ODDS_DEFAULTS,
    )
    row.update(overrides)
    return row


def _game_row(game_pk, date, home_team, away_team, home_score, away_score):
    return dict(
        game_pk=game_pk,
        game_date=pd.Timestamp(date),
        season=pd.Timestamp(date).year,
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
    )


def test_normalize_team_token_strips_whitespace_and_trailing_punctuation():
    assert _normalize_team_token("CIN") == "CIN"
    assert _normalize_team_token("CIN ") == "CIN"
    assert _normalize_team_token("CIN -") == "CIN"
    assert _normalize_team_token(" stl") == "STL"


def test_load_team_crosswalk_covers_known_franchise_variants():
    crosswalk = load_team_crosswalk(CROSSWALK_CONFIG_PATH)
    assert crosswalk["ARI"] == "AZ" and crosswalk["AZ"] == "AZ"
    assert crosswalk["OAK"] == "ATH" and crosswalk["ATH"] == "ATH"
    assert crosswalk["FLO"] == "MIA" and crosswalk["MIA"] == "MIA"
    assert crosswalk["SLC"] == "STL" and crosswalk["STL"] == "STL"
    assert crosswalk["CHW"] == "CWS"
    assert crosswalk["WAS"] == "WSH"
    assert "AL" not in crosswalk and "NL" not in crosswalk


def test_clean_unique_match_with_matching_score_is_accepted():
    betting = pd.DataFrame([_betting_row(1, "20240601", "PHI", "ATL", 3, 4)])
    games = pd.DataFrame([_game_row(9001, "2024-06-01", "ATL", "PHI", 4, 3)])

    accepted, review = build_betting_lines(betting, games)

    assert len(review) == 0
    assert len(accepted) == 1
    assert accepted.loc[0, "game_pk"] == 9001
    assert accepted.loc[0, "season"] == 2024
    assert accepted.loc[0, "total_open"] == 8.5


def test_unmapped_team_code_goes_to_review():
    betting = pd.DataFrame([_betting_row(1, "20240712", None, None, 5, 2, raw_away_team="AL", raw_home_team="NL")])
    games = pd.DataFrame([_game_row(9002, "2024-07-12", "NL", "AL", 2, 5)])

    accepted, review = build_betting_lines(betting, games)

    assert len(accepted) == 0
    assert review.loc[0, "reason"] == "unmapped team code"


def test_season_not_in_games_table_goes_to_review():
    betting = pd.DataFrame([_betting_row(1, "20110712", "ATL", "PHI", 3, 4)])
    games = pd.DataFrame([_game_row(9001, "2024-06-01", "ATL", "PHI", 4, 3)])  # only 2024 built

    accepted, review = build_betting_lines(betting, games)

    assert len(accepted) == 0
    assert review.loc[0, "reason"] == "season 2011 not present in games table"


def test_doubleheader_on_betting_side_goes_to_review_not_guessed():
    betting = pd.DataFrame([
        _betting_row(1, "20240601", "PHI", "ATL", 3, 4),
        _betting_row(2, "20240601", "PHI", "ATL", 1, 0),
    ])
    games = pd.DataFrame([_game_row(9001, "2024-06-01", "ATL", "PHI", 4, 3)])

    accepted, review = build_betting_lines(betting, games)

    assert len(accepted) == 0
    assert (review["reason"] == "doubleheader (multiple betting-line rows for this date/matchup)").all()
    assert len(review) == 2


def test_doubleheader_on_games_side_goes_to_review_not_guessed():
    betting = pd.DataFrame([_betting_row(1, "20240601", "PHI", "ATL", 3, 4)])
    games = pd.DataFrame([
        _game_row(9001, "2024-06-01", "ATL", "PHI", 4, 3),
        _game_row(9002, "2024-06-01", "ATL", "PHI", 1, 0),
    ])

    accepted, review = build_betting_lines(betting, games)

    assert len(accepted) == 0
    assert review.loc[0, "reason"] == "doubleheader (multiple games in games table for this date/matchup)"


def test_no_matching_game_goes_to_review():
    betting = pd.DataFrame([_betting_row(1, "20240601", "PHI", "ATL", 3, 4)])
    games = pd.DataFrame([_game_row(9001, "2024-06-02", "ATL", "PHI", 4, 3)])  # wrong date, same season

    accepted, review = build_betting_lines(betting, games)

    assert len(accepted) == 0
    assert review.loc[0, "reason"] == "no matching game found"


def test_score_mismatch_goes_to_review_instead_of_accepted():
    betting = pd.DataFrame([_betting_row(1, "20240601", "PHI", "ATL", 3, 4)])
    games = pd.DataFrame([_game_row(9001, "2024-06-01", "ATL", "PHI", 5, 3)])  # home_score disagrees: 5 vs 4

    accepted, review = build_betting_lines(betting, games)

    assert len(accepted) == 0
    assert review.loc[0, "reason"] == "score mismatch vs games table"


def test_american_odds_to_implied_prob_favorite_and_underdog():
    # favorite (negative) and underdog (positive) odds use different formulas
    implied = american_odds_to_implied_prob(np.array([-150, 130]))
    assert implied[0] == pytest.approx(0.6)
    assert implied[1] == pytest.approx(100.0 / 230.0)


def test_no_vig_home_win_prob_sums_to_one_across_both_sides():
    # -150/+130 has a raw overround (0.6 + 0.4348 = 1.0348, i.e. the vig);
    # normalizing both sides removes it so they sum to exactly 1.0.
    home_prob = no_vig_home_win_prob(np.array([-150]), np.array([130]))[0]
    away_prob = no_vig_home_win_prob(np.array([130]), np.array([-150]))[0]

    assert home_prob + away_prob == pytest.approx(1.0)
    assert home_prob == pytest.approx(0.6 / (0.6 + 100.0 / 230.0))


def test_no_vig_home_win_prob_is_symmetric_for_a_pick_em_game():
    # equal odds on both sides (a "pick 'em") should de-vig to exactly 0.5/0.5
    home_prob = no_vig_home_win_prob(np.array([-110]), np.array([-110]))[0]
    assert home_prob == pytest.approx(0.5)
