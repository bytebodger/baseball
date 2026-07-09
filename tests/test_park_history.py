import pandas as pd

from src.data.park_history import DEFAULT_CONFIG_PATH, load_park_history, resolve_park_id


def test_load_park_history_covers_known_relocations():
    park_history = load_park_history(DEFAULT_CONFIG_PATH)
    assert {"MIA", "ATL", "TEX", "ATH"} <= set(park_history.keys())


def test_team_without_a_relocation_uses_its_own_team_code_as_park_id():
    park_history = {"ATL": [{"from_season": 2015, "park_id": "ATL_TURNER_FIELD"}]}
    result = resolve_park_id(pd.Series(["NYY", "BOS"]), pd.Series([2018, 2022]), park_history)
    assert result.tolist() == ["NYY", "BOS"]


def test_relocation_switches_park_id_at_the_right_season():
    park_history = {
        "ATL": [
            {"from_season": 2015, "park_id": "ATL_TURNER_FIELD"},
            {"from_season": 2017, "park_id": "ATL_TRUIST_PARK"},
        ]
    }
    result = resolve_park_id(pd.Series(["ATL", "ATL", "ATL"]), pd.Series([2016, 2017, 2020]), park_history)
    assert result.tolist() == ["ATL_TURNER_FIELD", "ATL_TRUIST_PARK", "ATL_TRUIST_PARK"]


def test_real_park_history_resolves_all_known_relocations():
    park_history = load_park_history(DEFAULT_CONFIG_PATH)
    teams = pd.Series(["MIA", "MIA", "ATL", "ATL", "TEX", "TEX", "ATH", "ATH"])
    seasons = pd.Series([2011, 2012, 2016, 2017, 2019, 2020, 2024, 2025])
    result = resolve_park_id(teams, seasons, park_history)
    assert result.tolist() == [
        "MIA_SUN_LIFE_STADIUM", "MIA_MARLINS_PARK",
        "ATL_TURNER_FIELD", "ATL_TRUIST_PARK",
        "TEX_GLOBE_LIFE_PARK", "TEX_GLOBE_LIFE_FIELD",
        "ATH_OAKLAND_COLISEUM", "ATH_SUTTER_HEALTH_PARK",
    ]


def test_season_before_earliest_entry_falls_back_to_the_earliest_known_park():
    # 2005 predates this team's earliest listed entry (2017) -- assumed to
    # still be that same park, since it's the earliest one on record.
    park_history = {"ATL": [{"from_season": 2017, "park_id": "ATL_TRUIST_PARK"}]}
    result = resolve_park_id(pd.Series(["ATL"]), pd.Series([2005]), park_history)
    assert result.tolist() == ["ATL_TRUIST_PARK"]


def test_resolve_park_id_preserves_original_index():
    park_history = {}
    home_team = pd.Series(["NYY", "BOS"], index=[10, 20])
    season = pd.Series([2020, 2021], index=[10, 20])
    result = resolve_park_id(home_team, season, park_history)
    assert result.index.tolist() == [10, 20]
