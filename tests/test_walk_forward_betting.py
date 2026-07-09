import pandas as pd

from src.data.build_features import build_season_pitches_from_frame
from src.data.statcast_common import build_pitch_frame_from_raw, write_partitioned
from src.evaluation.betting_sim import DEFAULT_EDGE_THRESHOLD
from src.inference.backtest import ENSEMBLE_METHOD, GAME_PREDICTOR_METHOD, LOGISTIC_REGRESSION_METHOD
from src.inference.walk_forward_backtest import main as walk_forward_main
from tests.test_walk_forward_backtest import _write_encoder_checkpoint

GAMES_PER_SEASON = 2  # must be >= 2 with a mixed outcome: fit_stacking_ensemble needs both classes
# present in a fold's single validation season, unlike test_walk_forward_backtest's one-game-per-season fixture.


def _raw_row(pitcher, batter, game_date, at_bat_number, pitch_number, inning_topbot, home_team, away_team, home_score, away_score, game_pk, season):
    return {
        "pitcher": pitcher,
        "batter": batter,
        "game_date": game_date,
        "game_pk": game_pk,
        "game_year": season,
        "game_type": "R",
        "home_team": home_team,
        "away_team": away_team,
        "inning_topbot": inning_topbot,
        "inning": 1,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "pitch_type": "FF",
        "release_speed": 90.0,
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


def _write_multi_game_fixture(raw_dir, pitches_dir, seasons, games_per_season=GAMES_PER_SEASON):
    """`games_per_season` games per season (game_pk = season*10 + game index),
    alternating which side wins within the season."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for season in seasons:
        season_rows = []
        for g in range(games_per_season):
            game_pk = season * 10 + g
            date = f"{season}-04-0{g + 1}"
            home_score, away_score = (5, 3) if g % 2 == 0 else (3, 5)
            season_rows += [
                _raw_row(100, 101 + i, date, i + 1, 1, "Top", "DET", "CLE", home_score, away_score, game_pk, season)
                for i in range(9)
            ]
            season_rows += [
                _raw_row(200, 1 + i, date, 10 + i, 1, "Bot", "DET", "CLE", home_score, away_score, game_pk, season)
                for i in range(9)
            ]
        pd.DataFrame(season_rows).to_parquet(raw_dir / f"statcast_{season}.parquet")
        all_rows.extend(season_rows)

    raw_all = pd.DataFrame(all_rows)
    pitches = build_season_pitches_from_frame(build_pitch_frame_from_raw(raw_all))
    write_partitioned(pitches, pitches_dir)
    return pitches


def _write_open_close_betting_lines_fixture(betting_lines_dir, seasons, games_per_season=GAMES_PER_SEASON):
    """Every game priced as a -150/+130 home favorite at both open and close
    (no line movement) -- simple and deterministic, just needs to be a
    structurally valid betting_lines table for the integration test."""
    game_pks = [season * 10 + g for season in seasons for g in range(games_per_season)]
    df = pd.DataFrame(
        {
            "game_pk": game_pks,
            "season": [pk // 10 for pk in game_pks],
            "home_ml_open": [-150] * len(game_pks),
            "away_ml_open": [130] * len(game_pks),
            "home_ml_close": [-150] * len(game_pks),
            "away_ml_close": [130] * len(game_pks),
        }
    )
    df.to_parquet(betting_lines_dir, partition_cols=["season"], index=False)


def test_walk_forward_betting_runs_end_to_end_and_writes_results(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    seasons = [2015, 2016, 2017, 2018]
    pitches = _write_multi_game_fixture(raw_dir, pitches_dir, seasons)

    encoder_checkpoint = tmp_path / "player_encoder_best.pt"
    _write_encoder_checkpoint(encoder_checkpoint, pitches)

    training_config_path = tmp_path / "training_config.yaml"
    training_config_path.write_text(
        "hidden_dim: 16\n"
        "num_layers: 1\n"
        "dropout: 0.0\n"
        "runs_distribution: negative_binomial\n"
        "freeze_encoder: false\n"
        "training_mode: joint\n"
        "stage1_epochs: 1\n"
        "encoder_lr: 0.00001\n"
        "predictor_lr: 0.001\n"
    )

    checkpoint_dir = tmp_path / "checkpoints"
    games_dir = tmp_path / "games"
    pitcher_appearances_dir = tmp_path / "pitcher_appearances"
    batter_appearances_dir = tmp_path / "batter_appearances"
    betting_lines_dir = tmp_path / "betting_lines"
    _write_open_close_betting_lines_fixture(betting_lines_dir, seasons)

    # First, train the (tiny) fold checkpoints the same way the walk-forward
    # backtest itself does -- betting evaluation reuses them, never retrains.
    walk_forward_main(
        [
            "--season-range-start", "2015",
            "--season-range-end", "2018",
            "--train-years", "1",
            "--val-years", "1",
            "--test-years", "1",
            "--training-config", str(training_config_path),
            "--encoder-checkpoint", str(encoder_checkpoint),
            "--pitches-dir", str(pitches_dir),
            "--raw-dir", str(raw_dir),
            "--games-dir", str(games_dir),
            "--pitcher-appearances-dir", str(pitcher_appearances_dir),
            "--batter-appearances-dir", str(batter_appearances_dir),
            "--betting-lines-dir", str(betting_lines_dir),
            "--epochs", "1",
            "--patience", "1",
            "--batch-size", "4",
            "--device", "cpu",
            "--cache-dir", str(tmp_path / "sequence_cache"),
            "--checkpoint-dir", str(checkpoint_dir),
            "--output-dir", str(tmp_path / "reports_walk_forward"),
            "--n-resamples", "20",
            "--seed", "0",
        ]
    )

    from src.evaluation.walk_forward_betting import main as betting_main

    output_dir = tmp_path / "reports_betting"
    betting_main(
        [
            "--season-range-start", "2015",
            "--season-range-end", "2018",
            "--train-years", "1",
            "--val-years", "1",
            "--test-years", "1",
            "--pitches-dir", str(pitches_dir),
            "--raw-dir", str(raw_dir),
            "--games-dir", str(games_dir),
            "--pitcher-appearances-dir", str(pitcher_appearances_dir),
            "--batter-appearances-dir", str(batter_appearances_dir),
            "--betting-lines-dir", str(betting_lines_dir),
            "--batch-size", "4",
            "--device", "cpu",
            "--cache-dir", str(tmp_path / "sequence_cache"),
            "--checkpoint-dir", str(checkpoint_dir),
            "--output-dir", str(output_dir),
            "--edge-thresholds", str(DEFAULT_EDGE_THRESHOLD), "0.05",
            "--n-resamples", "20",
            "--seed", "0",
        ]
    )

    results = pd.read_csv(output_dir / "betting_sim_results.csv")
    assert set(results["fold"]) == {0, 1}
    assert set(results["method"]) == {GAME_PREDICTOR_METHOD, LOGISTIC_REGRESSION_METHOD, ENSEMBLE_METHOD}
    assert set(results["edge_threshold"]) == {DEFAULT_EDGE_THRESHOLD, 0.05}
    assert {"n_bets", "roi", "roi_ci_lo", "roi_ci_hi", "mean_clv", "clv_ci_lo", "clv_ci_hi"} <= set(results.columns)
    # every candidate game in this fixture has a matched betting line
    assert (results["n_candidates"] == GAMES_PER_SEASON).all()
    # one row per (fold, method, threshold): 2 folds x 3 methods x 2 thresholds
    assert len(results) == 2 * 3 * 2
