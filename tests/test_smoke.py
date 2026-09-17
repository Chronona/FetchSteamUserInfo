"""実際の Steam Web API を叩く契約テスト。

放置期間中に API の仕様が変わっていないか（必要なキーが今も返ってくるか）を
1 アカウント分のリクエストだけで確認する。

    pytest -m smoke

API キーが無い場合は自動で skip される。
"""

import pytest

import steam_api
from conftest import SAMPLE_UID

pytestmark = pytest.mark.smoke


def test_get_player_summary(real_api_key):
    player = steam_api.GetPlayerSummary(real_api_key, SAMPLE_UID)

    assert player["steamid"] == str(SAMPLE_UID)
    # 収集処理が前提にしているキー
    assert "personaname" in player


def test_get_player_summary_for_unused_id(real_api_key):
    # 17 桁に満たない ID は存在しないので players が空で返る
    assert steam_api.GetPlayerSummary(real_api_key, 1) == "Unused ID"


def test_get_player_summaries_handles_multiple_ids(real_api_key):
    players = steam_api.GetPlayerSummaries(
        real_api_key, steam_api.join_ids([SAMPLE_UID, 1])
    )

    # 実在するアカウントの分だけ返る
    assert [p["steamid"] for p in players] == [str(SAMPLE_UID)]


def test_get_owned_games(real_api_key):
    owned_games = steam_api.GetOwnedGames(real_api_key, SAMPLE_UID)

    assert owned_games["game_count"] > 0
    assert "appid" in owned_games["games"][0]


def test_get_recently_played_games(real_api_key):
    recently_played = steam_api.GetRecentlyPlayedGames(real_api_key, SAMPLE_UID, 0)

    assert "total_count" in recently_played


def test_get_steam_level(real_api_key):
    assert isinstance(steam_api.GetSteamLevel(real_api_key, SAMPLE_UID), int)


def test_get_badges(real_api_key):
    badges = steam_api.GetBadges(real_api_key, SAMPLE_UID)

    for key in ("badges", "player_xp", "player_level"):
        assert key in badges
    assert "badgeid" in badges["badges"][0]


def test_get_community_badge_progress(real_api_key):
    quests = steam_api.GetCommunityBadgeProgress(real_api_key, SAMPLE_UID, 2)

    assert "questid" in quests[0]
    assert isinstance(quests[0]["completed"], bool)


def test_is_playing_shared_game(real_api_key):
    # 例として Teardown の appid を使用
    result = steam_api.IsPlayingSharedGame(real_api_key, SAMPLE_UID, 1167630)

    assert "lender_steamid" in result


def test_fetch_user_detail(real_api_key):
    """Notebook の収集ループが 1 ユーザー分完走することの確認"""
    detail = steam_api.fetch_user_detail(real_api_key, SAMPLE_UID)

    assert detail["steamid"] == SAMPLE_UID
    assert detail["game_count"] > 0
    assert detail["cleared_quests"] >= 0
