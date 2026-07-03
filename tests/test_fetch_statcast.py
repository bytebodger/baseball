from datetime import date

import pandas as pd

from src.data import fetch_statcast


def test_month_chunks_full_season():
    chunks = list(fetch_statcast.month_chunks(2015, today=date(2015, 12, 31)))
    assert chunks[0] == ("2015-03-01", "2015-03-31")
    assert chunks[-1] == ("2015-11-01", "2015-11-30")
    assert len(chunks) == 9  # March through November


def test_month_chunks_capped_at_today():
    chunks = list(fetch_statcast.month_chunks(2026, today=date(2026, 7, 3)))
    assert chunks[-1] == ("2026-07-01", "2026-07-03")


def test_month_chunks_future_year_yields_nothing():
    chunks = list(fetch_statcast.month_chunks(2027, today=date(2026, 7, 3)))
    assert chunks == []


def test_fetch_season_skips_when_cached(tmp_path, monkeypatch):
    cached = tmp_path / "statcast_2019.parquet"
    cached.write_text("placeholder")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("statcast() should not be called when cache exists")

    monkeypatch.setattr(fetch_statcast, "statcast", fail_if_called)

    fetch_statcast.fetch_season(2019, raw_dir=tmp_path, force=False)

    assert cached.read_text() == "placeholder"


def test_fetch_season_downloads_and_concatenates(tmp_path, monkeypatch):
    calls = []

    def fake_statcast(start_dt, end_dt, **kwargs):
        calls.append((start_dt, end_dt))
        return pd.DataFrame({"pitch_type": ["FF"], "game_date": [start_dt]})

    monkeypatch.setattr(fetch_statcast, "statcast", fake_statcast)
    monkeypatch.setattr(fetch_statcast, "date", _FixedToday)

    fetch_statcast.fetch_season(2019, raw_dir=tmp_path, force=False)

    out_path = tmp_path / "statcast_2019.parquet"
    assert out_path.exists()
    saved = pd.read_parquet(out_path)
    assert len(saved) == len(calls)


class _FixedToday(date):
    @classmethod
    def today(cls):
        return date(2019, 12, 31)
