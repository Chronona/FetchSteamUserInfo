"""HTTP をスタブしたユニットテスト。実 API にはアクセスしない。"""

from urllib.parse import parse_qs, urlparse

import pytest
import requests
import responses

import steam_api


def stub(method_path, response_body, status=200):
    """https://api.steampowered.com/<method_path> への GET を登録する"""
    responses.add(
        responses.GET,
        "{}/{}".format(steam_api.STEAM_API_BASE, method_path),
        json=response_body,
        status=status,
    )


def sent_params():
    """直近のリクエストのクエリパラメータを dict で返す"""
    query = urlparse(responses.calls[-1].request.url).query
    return {k: v[0] for k, v in parse_qs(query).items()}


############################################################
# steam_get
############################################################


@responses.activate
def test_steam_get_builds_url_and_params(api_key):
    stub("ISteamUser/GetPlayerSummaries/v2/", {"response": {"players": []}})

    steam_api.steam_get(
        "ISteamUser", "GetPlayerSummaries", 2, api_key, steamids="12345"
    )

    request = responses.calls[-1].request
    assert urlparse(request.url).path == "/ISteamUser/GetPlayerSummaries/v2/"
    assert sent_params() == {
        "key": api_key,
        "format": "json",
        "steamids": "12345",
    }


@responses.activate
def test_steam_get_uses_https(api_key):
    stub("IPlayerService/GetOwnedGames/v1/", {"response": {}})

    steam_api.GetOwnedGames(api_key, 1)

    assert responses.calls[-1].request.url.startswith("https://")


@responses.activate
def test_steam_get_raises_on_error_status(api_key):
    # 不正な API キーなどでは 403 が返る。JSON デコード時ではなくここで気付きたい
    stub("ISteamUser/GetPlayerSummaries/v2/", {}, status=403)

    with pytest.raises(requests.HTTPError):
        steam_api.GetPlayerSummary(api_key, 1)


############################################################
# ISteamUser
############################################################


@responses.activate
def test_get_player_summary_returns_first_player(api_key):
    player = {"steamid": "76561198023414915", "personaname": "St4ck"}
    stub("ISteamUser/GetPlayerSummaries/v2/", {"response": {"players": [player]}})

    assert steam_api.GetPlayerSummary(api_key, player["steamid"]) == player


@responses.activate
def test_get_player_summary_reports_unused_id(api_key):
    stub("ISteamUser/GetPlayerSummaries/v2/", {"response": {"players": []}})

    assert steam_api.GetPlayerSummary(api_key, 1) == "Unused ID"


@responses.activate
def test_get_player_summaries_returns_all_players(api_key):
    players = [{"steamid": str(i)} for i in range(3)]
    stub("ISteamUser/GetPlayerSummaries/v2/", {"response": {"players": players}})

    assert steam_api.GetPlayerSummaries(api_key, "0,1,2") == players
    assert sent_params()["steamids"] == "0,1,2"


############################################################
# IPlayerService
############################################################


@responses.activate
def test_get_owned_games(api_key):
    body = {"game_count": 2, "games": [{"appid": 10}, {"appid": 80}]}
    stub("IPlayerService/GetOwnedGames/v1/", {"response": body})

    assert steam_api.GetOwnedGames(api_key, 1) == body


@responses.activate
def test_get_recently_played_games_passes_count(api_key):
    stub("IPlayerService/GetRecentlyPlayedGames/v1/", {"response": {"total_count": 0}})

    steam_api.GetRecentlyPlayedGames(api_key, 1, 0)

    assert sent_params()["count"] == "0"


@responses.activate
def test_get_steam_level_unwraps_value(api_key):
    stub("IPlayerService/GetSteamLevel/v1/", {"response": {"player_level": 5000}})

    assert steam_api.GetSteamLevel(api_key, 1) == 5000


@responses.activate
def test_get_badges(api_key):
    body = {"badges": [{"badgeid": 48}], "player_xp": 125265813, "player_level": 5000}
    stub("IPlayerService/GetBadges/v1/", {"response": body})

    assert steam_api.GetBadges(api_key, 1) == body


@responses.activate
def test_get_community_badge_progress_unwraps_quests(api_key):
    quests = [{"questid": 115, "completed": True}]
    stub(
        "IPlayerService/GetCommunityBadgeProgress/v1/", {"response": {"quests": quests}}
    )

    assert steam_api.GetCommunityBadgeProgress(api_key, 1, 2) == quests
    assert sent_params()["badgeid"] == "2"


@responses.activate
def test_is_playing_shared_game_passes_appid(api_key):
    stub(
        "IPlayerService/IsPlayingSharedGame/v1/", {"response": {"lender_steamid": "0"}}
    )

    result = steam_api.IsPlayingSharedGame(api_key, 1, 1167630)

    assert result == {"lender_steamid": "0"}
    assert sent_params()["appid_playing"] == "1167630"


############################################################
# 収集用ヘルパー
############################################################


def test_join_ids():
    assert steam_api.join_ids([1, 2, 3]) == "1,2,3"


def test_summarize_player_keeps_required_columns():
    player = {
        "steamid": "76561198023414915",
        "personaname": "St4ck",
        "timecreated": 1234567890,
        "avatar": "使わない項目",
    }

    assert steam_api.summarize_player(player) == {
        "steamid": "76561198023414915",
        "personaname": "St4ck",
        "timecreated": 1234567890,
    }


def test_summarize_player_fills_missing_timecreated():
    # 作成日を公開していないユーザーでも KeyError にならないこと
    player = {"steamid": "1", "personaname": "no timecreated"}

    assert steam_api.summarize_player(player)["timecreated"] == ""


@responses.activate
def test_count_existing_accounts_samples_within_range(api_key):
    players = [{"steamid": str(i)} for i in range(62)]
    stub("ISteamUser/GetPlayerSummaries/v2/", {"response": {"players": players}})

    count = steam_api.count_existing_accounts(
        api_key, 76561198000000000, 76561199000000000, sample_size=100
    )

    assert count == 62
    sampled = [int(i) for i in sent_params()["steamids"].split(",")]
    assert len(sampled) == 100
    assert len(set(sampled)) == 100  # 重複なし
    assert all(76561198000000000 <= i < 76561199000000000 for i in sampled)


@responses.activate
def test_fetch_user_detail_collects_games_and_badges(api_key):
    stub(
        "IPlayerService/GetOwnedGames/v1/",
        {"response": {"game_count": 1, "games": [{"appid": 10}]}},
    )
    stub(
        "IPlayerService/GetBadges/v1/",
        {
            "response": {
                "badges": [{"badgeid": 48}],
                "player_xp": 100,
                "player_level": 5,
            }
        },
    )
    stub(
        "IPlayerService/GetCommunityBadgeProgress/v1/",
        {
            "response": {
                "quests": [
                    {"questid": 115, "completed": True},
                    {"questid": 128, "completed": True},
                    {"questid": 134, "completed": False},
                ]
            }
        },
    )

    detail = steam_api.fetch_user_detail(api_key, 76561198023414915)

    assert detail["game_count"] == 1
    assert detail["player_level"] == 5
    assert detail["cleared_quests"] == 2  # completed が True のものだけ数える


@responses.activate
def test_fetch_user_detail_fills_none_for_private_profile(api_key):
    # 非公開プロフィールでは response が空で返る
    stub("IPlayerService/GetOwnedGames/v1/", {"response": {}})
    stub("IPlayerService/GetBadges/v1/", {"response": {}})

    detail = steam_api.fetch_user_detail(api_key, 1)

    assert detail["steamid"] == 1
    assert detail["game_count"] is None
    assert detail["games"] is None
    assert detail["badges"] is None
    assert detail["cleared_quests"] is None
    # バッジが取れない場合はコミュニティバッジ進捗を呼びに行かない
    assert len(responses.calls) == 2
