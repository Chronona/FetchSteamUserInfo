"""Steam Web API の薄いラッパー。

各関数名は Steam Web API のメソッド名にそのまま対応させている
（PEP 8 の命名とは異なるが、公式ドキュメントとの対応を優先）。

- https://partner.steamgames.com/doc/webapi
- https://partner.steamgames.com/doc/webapi/ISteamUser
- https://partner.steamgames.com/doc/webapi/IPlayerService
"""

import random

import requests

STEAM_API_BASE = "https://api.steampowered.com"
DEFAULT_TIMEOUT = 30

# GetPlayerSummaries が 1 リクエストで扱える steamid の上限
MAX_IDS_PER_REQUEST = 100


def steam_get(interface, method, version, api_key, timeout=DEFAULT_TIMEOUT, **params):
    """Steam Web API を呼び出し、JSON の response 部分を返す。

    例: steam_get("ISteamUser", "GetPlayerSummaries", 2, api_key, steamids=uid)
    """
    url = "{}/{}/{}/v{}/".format(STEAM_API_BASE, interface, method, version)
    r = requests.get(
        url, params=dict(key=api_key, format="json", **params), timeout=timeout
    )
    # 4xx / 5xx をここで検知する（JSON デコード時の分かりにくいエラーを避ける）
    r.raise_for_status()
    return r.json()["response"]


############################################################
# ISteamUser Interface（一部）
############################################################


def GetPlayerSummary(api_key, uid):
    """アカウント情報取得（1 アカウントだけ。存在しない id を入力しても OK）"""
    players = steam_get("ISteamUser", "GetPlayerSummaries", 2, api_key, steamids=uid)[
        "players"
    ]
    # ユーザー情報が返って来たかで ID の使用/未使用をチェック
    if not players:
        return "Unused ID"
    return players[0]


def GetPlayerSummaries(api_key, uids):
    """アカウント情報取得（複数アカウント。存在が確認されているアカウントのみ）

    uids は MAX_IDS_PER_REQUEST 件までコンマ区切りで一気に検索できる。
    """
    return steam_get("ISteamUser", "GetPlayerSummaries", 2, api_key, steamids=uids)[
        "players"
    ]


############################################################
# IPlayerService Interface
############################################################


def GetOwnedGames(api_key, uid):
    """UID を使って所持ゲーム情報を取得"""
    return steam_get("IPlayerService", "GetOwnedGames", 1, api_key, steamid=uid)


def GetRecentlyPlayedGames(api_key, uid, count):
    """最近遊んだゲームを表示（すべて表示したい場合は count=0）"""
    return steam_get(
        "IPlayerService", "GetRecentlyPlayedGames", 1, api_key, steamid=uid, count=count
    )


def GetSteamLevel(api_key, uid):
    """レベルの表示"""
    return steam_get("IPlayerService", "GetSteamLevel", 1, api_key, steamid=uid)[
        "player_level"
    ]


def GetBadges(api_key, uid):
    """所持バッジ"""
    return steam_get("IPlayerService", "GetBadges", 1, api_key, steamid=uid)


def GetCommunityBadgeProgress(api_key, uid, bid):
    """コミュニティバッジ進捗"""
    return steam_get(
        "IPlayerService",
        "GetCommunityBadgeProgress",
        1,
        api_key,
        steamid=uid,
        badgeid=bid,
    )["quests"]


def IsPlayingSharedGame(api_key, uid, appid_playing):
    """ゲームを貸し出している場合に割り振られる SteamID を表示"""
    return steam_get(
        "IPlayerService",
        "IsPlayingSharedGame",
        1,
        api_key,
        steamid=uid,
        appid_playing=appid_playing,
    )


############################################################
# 収集用のヘルパー
############################################################


def join_ids(ids):
    """steamid のリストをリクエスト用のコンマ区切り文字列にする"""
    return ",".join(str(i) for i in ids)


def count_existing_accounts(api_key, start, stop, sample_size=MAX_IDS_PER_REQUEST):
    """[start, stop) から sample_size 件の ID を抽出し、実在した件数を返す"""
    # range をそのまま渡すことで、10 億件をリスト化せずにサンプリングできる
    ids = random.sample(range(start, stop), sample_size)
    return len(GetPlayerSummaries(api_key, join_ids(ids)))


def summarize_player(player):
    """GetPlayerSummaries の 1 件から CSV に落とす項目だけを取り出す"""
    return {
        "steamid": player["steamid"],
        "personaname": player["personaname"],
        # 作成日を公開していないユーザーがいるので既定値を入れておく
        "timecreated": player.get("timecreated", ""),
    }


def fetch_user_detail(api_key, uid):
    """1 ユーザー分の所持ゲーム・バッジ情報をまとめて取得する"""
    user_info = {"steamid": uid}

    # 所持ゲーム
    owned_games = GetOwnedGames(api_key, uid)
    user_info["game_count"] = owned_games.get("game_count")
    user_info["games"] = owned_games.get("games")

    # バッジ情報
    badges = GetBadges(api_key, uid)
    if "badges" in badges:
        user_info["badges"] = badges["badges"]
        user_info["player_xp"] = badges["player_xp"]
        user_info["player_level"] = badges["player_level"]
        # コミュニティバッジ進捗
        quests = GetCommunityBadgeProgress(api_key, uid, 2)
        user_info["cleared_quests"] = sum(1 for q in quests if q["completed"])
        user_info["quests"] = quests
    else:
        # プロフィールを非表示設定にしている場合は取得できない
        user_info.update(
            badges=None,
            player_xp=None,
            player_level=None,
            cleared_quests=None,
            quests=None,
        )

    return user_info
