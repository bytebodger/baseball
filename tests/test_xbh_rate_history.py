import pandas as pd
import pytest

from src.data.xbh_rate_history import build_xbh_rate_history, xbh_rate_for, xbh_rate_features_batch


def _synthetic_batted_balls() -> pd.DataFrame:
    # pitcher 100: single (Jan, not XBH), double (Feb, XBH), field_out (Mar, not XBH).
    # pitcher 200: home_run (mid-Jan, XBH).
    return pd.DataFrame(
        {
            "pitcher_id": [100, 100, 100, 200],
            "batter_id": [900, 900, 901, 902],
            "game_date": pd.to_datetime(["2023-03-01", "2023-01-01", "2023-02-01", "2023-01-15"]),
            "outcome": ["field_out", "single", "double", "home_run"],
        }
    )


def test_build_xbh_rate_history_sorts_each_players_events_by_date():
    history = build_xbh_rate_history(_synthetic_batted_balls(), "pitcher_id")
    dates = history.dates_by_player[100]
    assert list(dates) == sorted(dates)
    # Jan (single, 0.0), Feb (double, 1.0), Mar (field_out, 0.0) order.
    assert history.is_xbh_by_player[100].tolist() == pytest.approx([0.0, 1.0, 0.0])


def test_build_xbh_rate_history_league_average_covers_every_row():
    history = build_xbh_rate_history(_synthetic_batted_balls(), "pitcher_id")
    # 2 XBH (pitcher 100's double, pitcher 200's home run) out of 4 total rows.
    assert history.league_avg_xbh_rate == pytest.approx(2 / 4)


def test_xbh_rate_for_uses_only_strictly_prior_events():
    history = build_xbh_rate_history(_synthetic_batted_balls(), "pitcher_id")
    jan_ns = pd.Timestamp("2023-01-01").value
    feb_ns = pd.Timestamp("2023-02-01").value
    late_ns = pd.Timestamp("2023-06-01").value

    # Querying exactly at pitcher 100's first event date excludes that same-day event.
    rate = xbh_rate_for(history, 100, jan_ns, min_events=0)
    assert rate == pytest.approx(history.league_avg_xbh_rate)

    # As of Feb 1 (exclusive): only the Jan 1 single (not XBH) is prior.
    rate = xbh_rate_for(history, 100, feb_ns, min_events=0)
    assert rate == pytest.approx(0.0)

    # Well after every real event: single + double + field_out -> 1/3 XBH.
    rate = xbh_rate_for(history, 100, late_ns, min_events=0)
    assert rate == pytest.approx(1 / 3)


def test_xbh_rate_for_falls_back_to_league_average_below_min_events():
    history = build_xbh_rate_history(_synthetic_batted_balls(), "pitcher_id")
    late_ns = pd.Timestamp("2023-06-01").value
    rate = xbh_rate_for(history, 100, late_ns, min_events=5)
    assert rate == pytest.approx(history.league_avg_xbh_rate)
    rate = xbh_rate_for(history, 100, late_ns, min_events=3)
    assert rate == pytest.approx(1 / 3)


def test_xbh_rate_for_falls_back_to_league_average_for_unknown_player():
    history = build_xbh_rate_history(_synthetic_batted_balls(), "pitcher_id")
    rate = xbh_rate_for(history, 999999, pd.Timestamp("2023-06-01").value)
    assert rate == pytest.approx(history.league_avg_xbh_rate)


def test_xbh_rate_features_batch_matches_single_lookup_row_by_row():
    history = build_xbh_rate_history(_synthetic_batted_balls(), "pitcher_id")
    player_ids = pd.Series([100, 100, 200, 999999])
    game_dates = pd.Series(pd.to_datetime(["2023-02-15", "2023-06-01", "2023-06-01", "2023-06-01"]))

    batch_result = xbh_rate_features_batch(history, player_ids, game_dates, min_events=0)
    for i, (pid, date) in enumerate(zip(player_ids, game_dates)):
        expected = xbh_rate_for(history, int(pid), pd.Timestamp(date).value, min_events=0)
        assert batch_result[i] == pytest.approx(expected)
