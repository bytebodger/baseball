import numpy as np
import pandas as pd
import torch

from src.data.game_dataset import (
    GameOutcomeDataset,
    _build_season_game_tables,
    ensure_game_tables_built,
    load_game_split,
)
from src.data.sequence_dataset import PlayerPitchSequenceDataset
from src.data.statcast_common import build_pitch_frame_from_raw
from src.data.build_features import build_season_pitches_from_frame


def _row(game_pk, game_date, season, home_team, away_team, inning, topbot, at_bat, pitch, pitcher, batter, home_score, away_score, game_type="R"):
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "game_year": season,
        "game_type": game_type,
        "home_team": home_team,
        "away_team": away_team,
        "inning_topbot": topbot,
        "inning": inning,
        "at_bat_number": at_bat,
        "pitch_number": pitch,
        "pitcher": pitcher,
        "batter": batter,
        "post_home_score": home_score,
        "post_away_score": away_score,
    }


def _fake_season_raw() -> pd.DataFrame:
    """Three games, season 2023, all DET home games:

    game 1 (2023-04-01): DET starter=100 (vs away batters 101-110, 10 distinct
      -- the 10th is a "pinch hitter" appearing only after the first 9).
      DET batters 1-10 bat in the bottom half (same pinch-hitter setup).
      Final score DET 5, CLE 3 (DET wins).

    game 3 (2023-04-10): DET starter=100 again (9-day rest). Pitcher 150
      relieves for DET later in the game (is_starter=False), within the
      bullpen trailing window of game 2.

    game 2 (2023-04-15): DET starter=100 again (5-day rest from game 3).
      DET wins 2-1. Pitcher 150's game-3 relief appearance (5 days prior)
      should show up in game 2's home bullpen; nothing from game 1 (14+
      days prior) should.
    """
    rows = []

    # --- game 1 ---
    for i in range(10):
        rows.append(_row(1, "2023-04-01", 2023, "DET", "CLE", 1, "Top", i + 1, 1, 100, 101 + i, 5, 3))
    for i in range(10):
        rows.append(_row(1, "2023-04-01", 2023, "DET", "CLE", 1, "Bot", 11 + i, 1, 200, 1 + i, 5, 3))

    # --- game 3 ---
    rows.append(_row(3, "2023-04-10", 2023, "DET", "BOS", 1, "Top", 1, 1, 100, 301, 4, 2))
    rows.append(_row(3, "2023-04-10", 2023, "DET", "BOS", 1, "Top", 2, 1, 100, 302, 4, 2))
    rows.append(_row(3, "2023-04-10", 2023, "DET", "BOS", 2, "Top", 3, 1, 150, 303, 4, 2))  # DET reliever
    rows.append(_row(3, "2023-04-10", 2023, "DET", "BOS", 1, "Bot", 4, 1, 310, 1, 4, 2))

    # --- game 2 ---
    rows.append(_row(2, "2023-04-15", 2023, "DET", "BOS", 1, "Top", 1, 1, 100, 301, 2, 1))
    rows.append(_row(2, "2023-04-15", 2023, "DET", "BOS", 1, "Bot", 2, 1, 320, 1, 2, 1))

    return pd.DataFrame(rows)


def test_starters_identified_from_first_pitch_and_score_home_win_month(tmp_path):
    raw_path = tmp_path / "statcast_2023.parquet"
    _fake_season_raw().to_parquet(raw_path)

    games, pitcher_appearances, batter_appearances = _build_season_game_tables(raw_path)

    game1 = games[games["game_pk"] == 1].iloc[0]
    # Top-half rows = away team batting = home team (DET) pitching, so
    # pitcher 100 (all Top-half rows) is DET's starter; pitcher 200
    # (all Bot-half rows) is CLE's.
    assert game1["home_starter_id"] == 100
    assert game1["away_starter_id"] == 200
    assert game1["home_score"] == 5 and game1["away_score"] == 3
    assert bool(game1["home_win"]) is True
    assert game1["month"] == 4


def test_home_starter_is_the_top_half_pitcher():
    # Sanity-check the inning_topbot -> team assignment directly: Top means
    # the away team bats, so the pitcher in Top-half rows pitches for the HOME team.
    raw = _fake_season_raw()
    game1_top = raw[(raw["game_pk"] == 1) & (raw["inning_topbot"] == "Top")]
    assert (game1_top["pitcher"] == 100).all()  # pitcher 100 -> home (DET) starter


def test_rest_days_computed_between_consecutive_starts(tmp_path):
    raw_path = tmp_path / "statcast_2023.parquet"
    _fake_season_raw().to_parquet(raw_path)
    games, _, _ = _build_season_game_tables(raw_path)

    game1 = games[games["game_pk"] == 1].iloc[0]
    game3 = games[games["game_pk"] == 3].iloc[0]
    game2 = games[games["game_pk"] == 2].iloc[0]

    assert pd.isna(game1["home_starter_rest_days"])  # pitcher 100's first-ever start
    assert game3["home_starter_rest_days"] == 9  # 04-10 minus 04-01
    assert game2["home_starter_rest_days"] == 5  # 04-15 minus 04-10


def test_pitcher_appearances_marks_reliever_as_non_starter(tmp_path):
    raw_path = tmp_path / "statcast_2023.parquet"
    _fake_season_raw().to_parquet(raw_path)
    _, pitcher_appearances, _ = _build_season_game_tables(raw_path)

    game3_det = pitcher_appearances[(pitcher_appearances["game_pk"] == 3) & (pitcher_appearances["team"] == "DET")]
    starter_row = game3_det[game3_det["pitcher_id"] == 100].iloc[0]
    reliever_row = game3_det[game3_det["pitcher_id"] == 150].iloc[0]
    assert bool(starter_row["is_starter"]) is True
    assert bool(reliever_row["is_starter"]) is False


def test_incomplete_game_is_dropped(tmp_path):
    raw = _fake_season_raw()
    # A 4th game with only a home-side pitcher (no away pitching rows at all),
    # so away_starter_id can never be resolved -- should be dropped, not crash.
    raw = pd.concat(
        [raw, pd.DataFrame([_row(4, "2023-04-20", 2023, "DET", "BOS", 1, "Top", 1, 1, 100, 301, 0, 0)])],
        ignore_index=True,
    )
    raw_path = tmp_path / "statcast_2023.parquet"
    raw.to_parquet(raw_path)

    games, _, _ = _build_season_game_tables(raw_path)
    assert 4 not in games["game_pk"].tolist()


def test_spring_training_games_are_excluded(tmp_path):
    raw = _fake_season_raw()
    raw = pd.concat(
        [
            raw,
            pd.DataFrame(
                [
                    _row(5, "2023-03-05", 2023, "DET", "BOS", 1, "Top", 1, 1, 100, 301, 1, 0, game_type="S"),
                    _row(5, "2023-03-05", 2023, "DET", "BOS", 1, "Bot", 2, 1, 320, 1, 1, 0, game_type="S"),
                ]
            ),
        ],
        ignore_index=True,
    )
    raw_path = tmp_path / "statcast_2023.parquet"
    raw.to_parquet(raw_path)

    games, pitcher_appearances, batter_appearances = _build_season_game_tables(raw_path)

    assert 5 not in games["game_pk"].tolist()
    assert 5 not in pitcher_appearances["game_pk"].tolist()
    assert 5 not in batter_appearances["game_pk"].tolist()


def _game_outcome_dataset_fixture(season=2023):
    game_date = pd.Timestamp(f"{season}-04-15")
    games = pd.DataFrame(
        {
            "game_pk": [2],
            "game_date": [game_date],
            "season": [season],
            "home_team": ["DET"],
            "away_team": ["BOS"],
            "home_score": [2],
            "away_score": [1],
            "home_win": [True],
            "month": [4],
            "home_starter_id": [100],
            "away_starter_id": [320],
            "home_starter_rest_days": [5.0],
            "away_starter_rest_days": [np.nan],
        }
    )
    pitcher_appearances = pd.DataFrame(
        {
            "game_pk": [1, 3, 3],
            "team": ["DET", "DET", "DET"],
            "pitcher_id": [100, 100, 150],
            "game_date": [pd.Timestamp("2023-04-01"), pd.Timestamp("2023-04-10"), pd.Timestamp("2023-04-10")],
            "season": [2023, 2023, 2023],
            "is_starter": [True, True, False],
        }
    )
    batter_appearances = pd.DataFrame(
        {
            "game_pk": [2] * 10,
            "team": ["DET"] * 10,
            "batter_id": list(range(1, 11)),
            "game_date": [pd.Timestamp("2023-04-15")] * 10,
            "season": [2023] * 10,
            "first_at_bat_number": list(range(1, 11)),
        }
    )
    return games, pitcher_appearances, batter_appearances


def _clean_pitches_for_dataset():
    """Real pipeline-shaped pitch history for pitcher 100 (starts) and batters
    1-10, spanning before and *on/after* the 2023-04-15 game date -- the
    on/after rows exist specifically to verify they get excluded (no leakage)."""
    rows = []
    for i, date in enumerate(["2023-03-01", "2023-04-01", "2023-04-10"]):
        rows.append(
            {
                "pitcher": 100, "batter": 1, "game_date": date, "game_pk": 900 + i,
                "at_bat_number": 1, "pitch_number": 1, "pitch_type": "FF",
                "release_speed": 90.0, "release_spin_rate": 2200, "spin_rate_deprecated": None,
                "plate_x": 0.1, "plate_z": 2.2, "balls": 0, "strikes": 0, "outs_when_up": 0,
                "on_1b": None, "on_2b": None, "on_3b": None, "home_score": 0, "away_score": 0, "n_thruorder_pitcher": 1,
                "inning": 1, "inning_topbot": "Top", "stand": "R", "p_throws": "L", "home_team": "DET", "away_team": "BOS",
                "game_year": 2023, "events": "field_out", "description": "hit_into_play",
            }
        )
    # A pitch strictly ON the game date and one strictly after it -- neither
    # should ever appear in the returned history for game_pk=2 (cutoff 04-15).
    for leak_date, leak_pk in [("2023-04-15", 2), ("2023-04-16", 999)]:
        rows.append(
            {
                "pitcher": 100, "batter": 1, "game_date": leak_date, "game_pk": leak_pk,
                "at_bat_number": 1, "pitch_number": 1, "pitch_type": "FF",
                "release_speed": 99.9, "release_spin_rate": 2200, "spin_rate_deprecated": None,
                "plate_x": 0.1, "plate_z": 2.2, "balls": 0, "strikes": 0, "outs_when_up": 0,
                "on_1b": None, "on_2b": None, "on_3b": None, "home_score": 0, "away_score": 0, "n_thruorder_pitcher": 1,
                "inning": 1, "inning_topbot": "Top", "stand": "R", "p_throws": "L", "home_team": "DET", "away_team": "BOS",
                "game_year": 2023, "events": "field_out", "description": "hit_into_play",
            }
        )
    return build_pitch_frame_from_raw(pd.DataFrame(rows))


def test_bullpen_respects_trailing_window_and_excludes_current_starter():
    games, pitcher_appearances, batter_appearances = _game_outcome_dataset_fixture()
    pitches = _clean_pitches_for_dataset()
    dataset = GameOutcomeDataset(pitches, games, pitcher_appearances, batter_appearances, max_seq_len=10)

    bullpen_ids = dataset._bullpen_ids("DET", pd.Timestamp("2023-04-15"), exclude_id=100)

    assert 150 in bullpen_ids  # relieved 5 days before, within the 14-day window
    assert 100 not in bullpen_ids  # excluded: this game's own starter


def test_bullpen_excludes_appearances_outside_the_window():
    games, pitcher_appearances, batter_appearances = _game_outcome_dataset_fixture()
    pitches = _clean_pitches_for_dataset()
    dataset = GameOutcomeDataset(
        pitches, games, pitcher_appearances, batter_appearances, max_seq_len=10, bullpen_window_days=14
    )

    # game_pk=1 (2023-04-01) is 14 days before 2023-04-15 -- right at the
    # boundary and, since only pitcher 100 (this game's starter) appeared
    # there, contributes nothing to the bullpen either way.
    bullpen_ids = dataset._bullpen_ids("DET", pd.Timestamp("2023-04-15"), exclude_id=100)
    assert bullpen_ids == [150]


def test_lineup_capped_to_max_size_and_ordered():
    games, pitcher_appearances, batter_appearances = _game_outcome_dataset_fixture()
    pitches = _clean_pitches_for_dataset()
    dataset = GameOutcomeDataset(
        pitches, games, pitcher_appearances, batter_appearances, max_seq_len=10, max_lineup_size=9
    )

    lineup_ids = dataset._lineup_ids(2, "DET")
    assert lineup_ids == list(range(1, 10))  # first 9 only, batter 10 (pinch hitter) excluded


def test_getitem_shape_and_no_leakage():
    games, pitcher_appearances, batter_appearances = _game_outcome_dataset_fixture()
    pitches = _clean_pitches_for_dataset()
    dataset = GameOutcomeDataset(pitches, games, pitcher_appearances, batter_appearances, max_seq_len=10)

    sample = dataset[0]

    assert sample["home_team"] == "DET" and sample["away_team"] == "BOS"
    assert sample["park_id"] == "DET"
    assert sample["month"] == 4
    assert sample["post_humidor"] is True  # fixture season is 2023, post-mandate
    assert sample["home_score"] == 2 and sample["away_score"] == 1
    assert sample["home_win"] is True
    assert sample["home_starter"]["has_history"] is True
    assert len(sample["home_lineup"]) == 9
    assert 150 in [pid for pid in dataset._bullpen_ids("DET", sample["game_date"], 100)]

    # No-leakage: the starter's history must only contain pitches strictly
    # before the game date, even though the fixture pitches include rows
    # dated exactly on and after it.
    starter_history_length = sample["home_starter"]["length"]
    assert starter_history_length == 3  # only the 3 pre-04-15 pitches


def test_post_humidor_flag_false_before_2022():
    games, pitcher_appearances, batter_appearances = _game_outcome_dataset_fixture(season=2019)
    pitches = _clean_pitches_for_dataset()
    dataset = GameOutcomeDataset(pitches, games, pitcher_appearances, batter_appearances, max_seq_len=10)

    assert dataset[0]["post_humidor"] is False


def test_warm_cache_makes_getitem_match_an_uncached_dataset(tmp_path):
    games, pitcher_appearances, batter_appearances = _game_outcome_dataset_fixture()
    pitches = _clean_pitches_for_dataset()
    cache_dir = tmp_path / "sequence_cache"

    warmed = GameOutcomeDataset(
        pitches, games, pitcher_appearances, batter_appearances, max_seq_len=10, cache_dir=cache_dir
    )
    pitcher_computed, batter_computed = warmed.warm_cache()
    assert pitcher_computed > 0  # at least the home/away starters
    assert (cache_dir / "pitcher").exists()

    uncached = GameOutcomeDataset(pitches, games, pitcher_appearances, batter_appearances, max_seq_len=10)
    cached_sample = warmed[0]
    uncached_sample = uncached[0]

    assert cached_sample["home_starter"]["length"] == uncached_sample["home_starter"]["length"]
    assert torch.allclose(cached_sample["home_starter"]["continuous"], uncached_sample["home_starter"]["continuous"])

    # a second warm_cache() call against the same games has nothing new to do
    pitcher_recomputed, batter_recomputed = warmed.warm_cache()
    assert pitcher_recomputed == 0
    assert batter_recomputed == 0


def test_ensure_game_tables_built_caches_partitions_to_disk(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _fake_season_raw().to_parquet(raw_dir / "statcast_2023.parquet")

    games_dir = tmp_path / "processed" / "games"
    pitcher_dir = tmp_path / "processed" / "pitcher_appearances"
    batter_dir = tmp_path / "processed" / "batter_appearances"

    ensure_game_tables_built([2023], raw_dir=raw_dir, games_dir=games_dir,
                             pitcher_appearances_dir=pitcher_dir, batter_appearances_dir=batter_dir)

    assert (games_dir / "season=2023").exists()
    assert (pitcher_dir / "season=2023").exists()
    assert (batter_dir / "season=2023").exists()


def test_load_game_split_train_val_season_overrides_and_appearance_season_end(tmp_path):
    """Walk-forward retraining needs (a) a non-default train/val season
    boundary and (b) pitcher/batter appearance history extended past that
    boundary into the test season (a workload/rest-day lookup for a
    test-season game needs that pitcher's own recent, also-test-season,
    appearances). Confirms both overrides actually change what's returned,
    not just that they're accepted."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    fake_2023 = _fake_season_raw()
    fake_2024 = _fake_season_raw()
    fake_2024["game_year"] = 2024
    fake_2024["game_date"] = fake_2024["game_date"].str.replace("2023", "2024")
    fake_2023.to_parquet(raw_dir / "statcast_2023.parquet")
    fake_2024.to_parquet(raw_dir / "statcast_2024.parquet")

    games_dir = tmp_path / "games"
    pitcher_dir = tmp_path / "pitcher_appearances"
    batter_dir = tmp_path / "batter_appearances"

    train_games, val_games, pitcher_appearances, batter_appearances = load_game_split(
        raw_dir=raw_dir, games_dir=games_dir, pitcher_appearances_dir=pitcher_dir, batter_appearances_dir=batter_dir,
        train_season_range=(2023, 2023), val_seasons=(2024,), appearance_season_end=2024,
    )

    assert set(train_games["season"].unique()) == {2023}
    assert set(val_games["season"].unique()) == {2024}
    # appearance_season_end=2024 (matching val here) means both seasons' appearances are present.
    assert set(pitcher_appearances["season"].unique()) == {2023, 2024}
    assert set(batter_appearances["season"].unique()) == {2023, 2024}


# --- Explicit no-leakage check: every pitch date actually used must be < the game date ---
#
# Sequence tensors returned by PlayerPitchSequenceDataset don't carry raw dates
# (only normalized continuous features + category indices), so to check real
# *dates* rather than just sequence lengths, each pitch's release_speed here
# encodes its own game_date (80 + days-since-epoch, monotonic and unique per
# date). Decoding the continuous tensor's release_speed column back through
# the dataset's own normalization stats recovers the actual dates used, which
# lets the test assert the no-leakage property directly against real returned
# tensors rather than re-deriving an "expected" filter and checking counts.
_EPOCH = pd.Timestamp("2023-01-01")


def _speed_for_date(date_str: str) -> float:
    return 80.0 + (pd.Timestamp(date_str) - _EPOCH).days


def _dated_row(game_pk, date_str, home_team, away_team, inning, topbot, at_bat, pitch, pitcher, batter, home_score=1, away_score=0, season=2023):
    return {
        "pitcher": pitcher,
        "batter": batter,
        "game_date": date_str,
        "game_pk": game_pk,
        "game_year": season,
        "game_type": "R",
        "home_team": home_team,
        "away_team": away_team,
        "inning_topbot": topbot,
        "inning": inning,
        "at_bat_number": at_bat,
        "pitch_number": pitch,
        "pitch_type": "FF",
        "release_speed": _speed_for_date(date_str),
        "release_spin_rate": 2200,
        "spin_rate_deprecated": None,
        "plate_x": 0.1,
        "plate_z": 2.2,
        "balls": 0,
        "strikes": 0,
        "outs_when_up": 0,
        "on_1b": None,
        "on_2b": None,
        "on_3b": None,
        "home_score": 0,
        "away_score": 0,
        "n_thruorder_pitcher": 1,
        "stand": "R",
        "p_throws": "L",
        "events": "field_out",
        "description": "hit_into_play",
        "post_home_score": home_score,
        "post_away_score": away_score,
    }


def _sample_of_games_raw() -> pd.DataFrame:
    """Three games for DET (starter 500 throughout, so his history carries
    across all three), a reliever (501) who appears for DET between games,
    and a 9-batter lineup (600-608) each game -- deliberately including
    pitches dated on/after each game's own date for the SAME players, so a
    pitch that's forbidden history for an earlier game is legitimate history
    for a later one and vice versa.
    """
    rows = []

    # Pre-history for starter 500, well before any of the three games.
    rows.append(_dated_row(1, "2023-03-01", "DET", "BOS", 1, "Top", 1, 1, 500, 700))
    rows.append(_dated_row(2, "2023-04-15", "DET", "BOS", 1, "Top", 1, 1, 500, 700))

    # game 10: 2023-05-01
    rows.append(_dated_row(10, "2023-05-01", "DET", "BOS", 1, "Top", 1, 1, 500, 701))
    for i in range(9):
        rows.append(_dated_row(10, "2023-05-01", "DET", "BOS", 1, "Bot", 2 + i, 1, 900, 600 + i))

    # Reliever 501 appears for DET between games 10 and 11 (within the
    # 14-day bullpen window of both 11 and 12).
    rows.append(_dated_row(6, "2023-05-04", "DET", "BOS", 2, "Top", 1, 1, 501, 702))

    # game 11: 2023-05-08
    rows.append(_dated_row(11, "2023-05-08", "DET", "BOS", 1, "Top", 1, 1, 500, 701))
    for i in range(9):
        rows.append(_dated_row(11, "2023-05-08", "DET", "BOS", 1, "Bot", 2 + i, 1, 901, 600 + i))

    # game 12: 2023-05-15
    rows.append(_dated_row(12, "2023-05-15", "DET", "BOS", 1, "Top", 1, 1, 500, 701))
    for i in range(9):
        rows.append(_dated_row(12, "2023-05-15", "DET", "BOS", 1, "Bot", 2 + i, 1, 902, 600 + i))

    return pd.DataFrame(rows)


def test_no_pitch_used_in_any_players_history_is_on_or_after_the_game_date(tmp_path):
    raw = _sample_of_games_raw()
    raw_path = tmp_path / "statcast_2023.parquet"
    raw.to_parquet(raw_path)

    games, pitcher_appearances, batter_appearances = _build_season_game_tables(raw_path)
    pitches = build_pitch_frame_from_raw(raw)

    dataset = GameOutcomeDataset(pitches, games, pitcher_appearances, batter_appearances, max_seq_len=20)

    mean, std = dataset.pitcher_sequences.continuous_stats["release_speed"]

    def _decode_dates(sequence: dict) -> list[pd.Timestamp]:
        if not sequence["has_history"]:
            return []
        release_speed = sequence["continuous"][:, 0] * std + mean
        return [_EPOCH + pd.Timedelta(days=round(v.item() - 80.0)) for v in release_speed]

    assert len(dataset) == 3  # the full "sample of games"

    checked_any = False
    for idx in range(len(dataset)):
        sample = dataset[idx]
        cutoff = sample["game_date"]

        player_sequences = (
            [sample["home_starter"], sample["away_starter"]]
            + sample["home_bullpen"]
            + sample["away_bullpen"]
            + sample["home_lineup"]
            + sample["away_lineup"]
        )
        for sequence in player_sequences:
            dates_used = _decode_dates(sequence)
            for date_used in dates_used:
                checked_any = True
                assert date_used < cutoff, (
                    f"game {sample['game_pk']} (cutoff {cutoff}): found a pitch dated {date_used}, "
                    "which is not strictly before the game being predicted"
                )

    assert checked_any  # sanity: the test actually exercised some non-empty histories
