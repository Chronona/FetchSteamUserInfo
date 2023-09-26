# steamapiを使ってみる
#%%
import json
import time
import requests
import random

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

#%%
# 自分用
with open(".ignore/mydata/steam_api.txt", "r") as f:
    api_key = f.read()

with open(".ignore/mydata/account.txt", "r") as f2:
    uid = f2.read()

#%%
# apikeyと使用するsteamIDを定義
# apikey = ""
# uid = ""

#%%
# St4ck氏のアカウント情報を取得する
uid = 76561198023414915


#%%
############################################################
# ISteamUser Interface（一部）
# https://partner.steamgames.com/doc/webapi/ISteamUser
############################################################

# アカウント情報取得（１アカウントだけ。存在しないidを入力してもOK）
def GetPlayerSummary(api_key, uid):
    url = "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/?key={}&steamids={}&format=json".format(
        api_key, uid
    )
    r = requests.get(url)
    # 結果はJSON形式なのでデコード
    data = json.loads(r.text)
    # ユーザー情報が帰って来たかでIDの使用/未使用をチェック
    if data["response"]["players"] == []:
        return "Unused ID"
    else:
        return data["response"]["players"][0]


# アカウント情報取得（複数アカウント。存在が確認されているアカウントのみ）
def GetPlayerSummaries(api_key, uids):  # steamidは100アカウントまで一気に検索可能（コンマ区切りで並べる）
    url = "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/?key={}&steamids={}&format=json".format(
        api_key, uids
    )
    r = requests.get(url)
    data = json.loads(r.text)
    return data["response"]["players"]


#%%
############################################################
# IPlayerService Interface
# https://partner.steamgames.com/doc/webapi/IPlayerService
############################################################

# UIDを使って所持ゲーム情報を取得
def GetOwnedGames(api_key, uid):
    url = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={}&steamid={}&format=json".format(
        api_key, uid
    )
    r = requests.get(url)
    # 結果はJSON形式なのでデコード
    data = json.loads(r.text)
    return data["response"]


# 最近遊んだゲームを表示
def GetRecentlyPlayedGames(api_key, uid, count):
    # 最近遊んだゲームをすべて表示したい場合はcount=0
    url = "https://api.steampowered.com/IPlayerService/GetRecentlyPlayedGames/v1/?key={}&steamid={}&count={}&format=json".format(
        api_key, uid, count
    )
    r = requests.get(url)
    data = json.loads(r.text)
    return data["response"]


# レベルの表示
def GetSteamLevel(api_key, uid):
    url = "https://api.steampowered.com/IPlayerService/GetSteamLevel/v1/?key={}&steamid={}&format=json".format(
        api_key, uid
    )
    r = requests.get(url)
    data = json.loads(r.text)
    return data["response"]["player_level"]


# 所持バッジ
def GetBadges(api_key, uid):
    url = "https://api.steampowered.com/IPlayerService/GetBadges/v1/?key={}&steamid={}&format=json".format(
        api_key, uid
    )
    r = requests.get(url)
    data = json.loads(r.text)
    return data["response"]


# コミュニティバッジ進捗
def GetCommunityBadgeProgress(api_key, uid, bid):
    url = "https://api.steampowered.com/IPlayerService/GetCommunityBadgeProgress/v1/?key={}&steamid={}&badgeid={}&format=json".format(
        api_key, uid, bid
    )
    r = requests.get(url)
    data = json.loads(r.text)
    return data["response"]["quests"]


# ゲームを貸し出している場合に割り振られるSteamIDを表示
def IsPlayingSharedGame(api_key, uid, appid_playing):
    url = "https://api.steampowered.com/IPlayerService/IsPlayingSharedGame/v1/?key={}&steamid={}&appid_playing={}&format=json".format(
        api_key, uid, appid_playing
    )
    r = requests.get(url)
    data = json.loads(r.text)
    return data["response"]


# %%
############################################################

# UIDを使って所持ゲーム情報を取得
ownd_games = GetOwnedGames(api_key, uid)
#%%
ownd_games["game_count"]
#%%
ownd_games["games"][:5]

#%%
# 最近遊んだゲームを表示
recently_played_games = GetRecentlyPlayedGames(api_key, uid, 0)
#%%
recently_played_games["total_count"]
#%%
recently_played_games["games"][:3]
# %%
# レベルの表示
steam_level = GetSteamLevel(api_key, uid)
steam_level
# %%
# 所持バッジ
badges = GetBadges(api_key, uid)
# %%
badges["badges"][:3]
# %%
badges["player_xp"]
# %%
badges["player_level"]
# %%
badges["player_xp_needed_to_level_up"]
# %%
badges["player_xp_needed_current_level"]
#%%
# コミュニティバッジ進捗（badge_id =2）
community_Badge_Progress = GetCommunityBadgeProgress(api_key, uid, 2)
community_Badge_Progress
# %%
# ゲームを貸し出している場合に割り振られるSteamIDを表示
# 例としてTeardownのappidを使用
IsPlayingSharedGame(api_key, uid, 1167630)

# %%
# steamidは17桁。いくつかのsteamアカウントをhttps://steamidfinder.com/で適当にしらべると、
# 初めの"76561198"は全員同じだった。この部分をsteamidを識別子として利用していると仮定して、
# 残り9桁を乱数で生成。
# 出鱈目なidから実在するアカウントを1万件探して情報を取得。

# %%
for i in tqdm(range(5)):
    # 76561198000000000 ~ 76561198999999999が格納されたジェネレータから候補ID抽出
    random_numbers = (i for i in range(76561198000000000, 76561199000000000))
    ids = random.sample(list(random_numbers), 100)
    result = GetPlayerSummaries(api_key, ids)
    print(len(result))
# 54, 65, 62, 71, 59。大体62.2%の確率で存在するIDがヒットするらしい。

# %%
#
for i in tqdm(range(5)):
    # 76561197000000000 ~ 76561197999999999が格納されたジェネレータから候補ID抽出
    random_numbers = (i for i in range(76561197000000000, 76561198000000000))
    ids = random.sample(list(random_numbers), 100)
    result = GetPlayerSummaries(api_key, ids)
    print(len(result))
# 3,2,1,2,2 ほとんどヒットしない
# %%
for i in tqdm(range(5)):
    # 76561199000000000 ~ 76561199999999999が格納されたジェネレータから候補ID抽出
    random_numbers = (i for i in range(76561199000000000, 76561200000000000))
    ids = random.sample(list(random_numbers), 100)
    result = GetPlayerSummaries(api_key, ids)
    print(len(result))
# 19, 14, 12, 18, 28。微妙

#%%
# 最低1万件標本が欲しいので多めに見積もって20000個乱数を生成する

# %%
# ID候補が入ったジェネレータ作成
random_numbers = (i for i in range(76561198000000000, 76561199000000000))
# 2万件抽出
ids = random.sample(list(random_numbers), 20000)
# 100件ずつしかリクエストできないので分割
ids_list = list(np.array_split(ids, 200))
ids_list
# %%
# 出力ファイル初期化
pd.DataFrame(
    {
        "steamid": [],
        "personaname": [],
        "timecreated": [],
    }
).to_csv("user_info.csv")

for i in tqdm(ids_list):
    # リクエスト用のID文字列作成
    ids_str = ",".join([str(j) for j in i])
    # ユーザー情報取得
    result = GetPlayerSummaries(api_key, ids_str)
    # ID, name, createtimeを取得して辞書化しリストに格納
    user_info_list = []
    for i in result:
        user_info = {}
        user_info["steamid"] = i["steamid"]
        user_info["personaname"] = i["personaname"]
        # 作成日がないユーザーに対しては例外処理
        try:
            user_info["timecreated"] = i["timecreated"]
        except KeyError:
            user_info["timecreated"] = ""
        user_info_list.append(user_info)

    # 出力
    # 大体60件ずつ追加されていることを確認したいのでindexは残す
    pd.DataFrame(user_info_list).to_csv("user_id.csv", mode="a", header=False)
# %%
# 11508件入手できた。概ね予想通り
# %%
# csv読み込み（indexは読み込まない）
df = pd.read_csv("user_id.csv", usecols=[1, 2, 3])
# timecreatedの値をUNIX timeに変換
df["timecreated"] = pd.to_datetime(df["timecreated"], utc=True, unit="s")
df

# %%
# ユーザー情報を取得
user_info_list = []
steam_ids = df["steamid"]
for id in tqdm(steam_ids):
    user_info = {}
    user_info["steamid"] = id

    # 情報が取得できないユーザーがいたので例外処理
    # プロフィールを非表示設定にしている場合は取得できない？

    # 所持ゲーム
    ownd_games = GetOwnedGames(api_key, id)
    try:
        user_info["game_count"] = ownd_games["game_count"]
        user_info["games"] = ownd_games["games"]
    except KeyError:
        user_info["game_count"] = None
        user_info["games"] = None

    # バッジ情報
    badges = GetBadges(api_key, id)
    try:
        user_info["badges"] = badges["badges"]
        user_info["player_xp"] = badges["player_xp"]
        user_info["player_level"] = badges["player_level"]
        # コミュニティバッジ進捗
        community_Badge_Progress = GetCommunityBadgeProgress(api_key, id, 2)
        user_info["cleard_quests"] = len(
            [i for i in community_Badge_Progress if i["completed"] == True]
        )
        user_info["quests"] = community_Badge_Progress
    except KeyError:
        user_info["badges"] = None
        user_info["player_xp"] = None
        user_info["player_level"] = None
        user_info["cleard_quests"] = None
        user_info["quests"] = None

    user_info_list.append(user_info)
    # ウェイト時間を設けておく
    time.sleep(0.5)

# DF化→jsonで出力
badge_df = pd.DataFrame(user_info_list)
badge_df.to_json("user_info.json")
# %%
GetOwnedGames(api_key, steam_ids[0])
# %%
badge_df = pd.DataFrame(user_info_list)
badge_df.to_json("user_info.json")
badge_df 
# %%
