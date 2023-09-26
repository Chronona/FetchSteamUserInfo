#%%
import collections

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import statsmodels.formula.api as sm

# グラフ描画時に文字化けするのでフォント指定
sns.set(font=["IPAGothic"])

#%%
df_in = pd.read_json("user_info.json")
df_in
#%%
# 所持ゲーム情報を公開しているユーザーは少ない
df_in.dropna(subset=["game_count"], how="any")
# 276

#%%
# バッジは比較的多数のユーザーが公開している
df_in.dropna(subset=["badges"], how="any")

#%%
# バッジ情報に欠損値があるデータは除外する
badge_df = df_in.dropna(subset=["badges"], how="any")

badge_df
#%%
# コミュティバッジ取得数が0のアカウントは、一般的なユーザーでないと判断し除外
# バッジ取得条件の１つに「ゲームをプレイする」があるぐらい簡単なクエストもあるので、
# クエスト達成数が0の場合は、アカウントを作っただけで利用していない可能性がある
badge_df = badge_df[badge_df["cleard_quests"] > 0].reset_index(drop=True)
badge_df


#%%
# 散布図
badge_df.plot.scatter(
    x="cleard_quests", y="player_xp", yticks=[10 ** i for i in range(7)], logy=True
)
# クエスト達成数が15を超えてくるとxpも増加傾向にある？

#%%
# 各ユーザのクエストの達成状況を調べてみる
quest_df = pd.DataFrame([i for i in badge_df["quests"]])
quest_ids = ["id_{}".format(quest_df.iloc[0, i]["questid"]) for i in quest_df]
quest_df.columns = quest_ids

# クエストを達成したかどうかのDF作成
is_completed = lambda x: x["completed"]
for i in quest_df.columns:
    # "completed"キーがTrueなら1,Falseなら0を格納
    quest_df[i] = [1 if i is True else 0 for i in quest_df[i].apply(is_completed)]

# 達成したクエストの合計値
quest_df["total"] = quest_df.apply(sum, axis=1)


quest_df


#%%
# 各クエストの達成ユーザー数を集計
quest_list = []
for i in quest_df.columns[:28]:
    completed_quests = {}
    completed_quests["id"] = i
    completed_quests["count"] = len(quest_df[quest_df[i] == True])
    quest_list.append(completed_quests)
quest_completed_df = pd.DataFrame(quest_list)
quest_completed_df
#%%
# 各クエストの達成条件を追加（steamのコミュニティバッジクエストページに書かれた条件を要約）
description = [
    "Steamガードを有効化",
    "携帯番号の追加",
    "Steamモバイルアプリを利用",
    "ブロードキャストを表示",
    "ストアディスカバリーキューを使用",
    "ウィッシュリストにゲームを追加",
    "フレンドリストにフレンドを追加",
    "ゲームをプレイ",
    "ゲームのレビューを書く",
    "スクリーンショットを投稿",
    "ムービーを投稿",
    "ワークショップのアイテムを評価",
    "ワークショップのアイテムをサブスクライブ",
    "Steamオーバーレイからガイドを表示",
    "コミュニティプロフィールにアバターを設定",
    "コミュニティプロフィールに本名を設定",
    "プロフィールの背景を設定",
    "グループに参加",
    "フレンドのプロフィールにコメント",
    "アクティビティフィード内のコンテンツを評価",
    "フレンドに向けてコメントを投稿",
    "フレンドのスクリーンショットにコメント",
    "ゲームバッジを作成",
    "プロフィール上にバッジを張り付け",
    "チャット内で絵文字を使用",
    "掲示板を検索",
    "コミュニテイマーケットを利用",
    "トレードを利用",
]

quest_completed_df["description"] = description
quest_completed_df
# %%
# 棒グラフ作成


x = quest_completed_df["description"]
y = quest_completed_df["count"][::-1]

fig = plt.figure(figsize=(10, 15))
ax = fig.add_subplot(1, 1, 1)
ax.barh(x, y, height=0.7, tick_label=x[::-1])

#%%
# クエストクリア数でヒストグラムを作成

y = quest_df["total"]

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.hist(y)

#%%
# https://www.it-swarm.jp.net/ja/python/scikitlearn%EF%BC%9A1%E6%AC%A1%E5%85%83%E9%85%8D%E5%88%97%E3%81%A7kmeans%E3%82%92%E5%AE%9F%E8%A1%8C%E3%81%99%E3%82%8B%E6%96%B9%E6%B3%95%E3%81%AF%EF%BC%9F/1051609236/
# のコードを借りて、クエストクリア数をクラスタリング
import numpy as np
import matplotlib.pyplot as plt


def get_jenks_breaks(data_list, number_class):
    data_list.sort()
    mat1 = []
    for i in range(len(data_list) + 1):
        temp = []
        for j in range(number_class + 1):
            temp.append(0)
        mat1.append(temp)
    mat2 = []
    for i in range(len(data_list) + 1):
        temp = []
        for j in range(number_class + 1):
            temp.append(0)
        mat2.append(temp)
    for i in range(1, number_class + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2, len(data_list) + 1):
            mat2[j][i] = float("inf")
    v = 0.0
    for l in range(2, len(data_list) + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = float(data_list[i3 - 1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, number_class + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v
    k = len(data_list)
    kclass = []
    for i in range(number_class + 1):
        kclass.append(min(data_list))
    kclass[number_class] = float(data_list[len(data_list) - 1])
    count_num = number_class
    while count_num >= 2:  # print "rank = " + str(mat1[k][count_num])
        idx = int((mat1[k][count_num]) - 2)
        # print "val = " + str(data_list[idx])
        kclass[count_num - 1] = data_list[idx]
        k = int((mat1[k][count_num] - 1))
        count_num -= 1
    return kclass


x = quest_df["total"].to_list()
breaks = get_jenks_breaks(x, 3)

for line in breaks:
    print(line)
    plt.plot([line for _ in range(len(x))], "k--")

plt.plot(x)
plt.grid(True)
plt.show()

# 1 7 16 28.0にラインが引かれたので、
# クエスト進捗度(a)を
# 1 < a ≦ 7 : low
# 1 < a ≦ 16 : mid
# 16 < a ≦ 28 : high
# と分けることにした

# %%
# quest_df にクエスト進捗度ラベルを付与

progress_level = []
progress_name = []
for i in quest_df["total"]:
    if 1 <= i <= 7:
        progress_level.append(0)
        progress_name.append("low")
    elif 7 < i <= 16:
        progress_level.append(1)
        progress_name.append("mid")
    elif 16 < i <= 28:
        progress_level.append(2)
        progress_name.append("high")
    # 念の為例外があれば表示
    else:
        print(i)
quest_df["progress_level"] = progress_level
quest_df["progress_name"] = progress_name

quest_df

#%%

# 決定木

from sklearn.datasets import load_iris
from sklearn import tree

x_data = quest_df.drop(columns=["total", "progress_level", "progress_name"])
x_data.columns = description
y_target = quest_df["progress_level"]
clf = tree.DecisionTreeClassifier(max_depth=3)  # 深さを指定
clf.fit(x_data, y_target)

viz = dtreeviz(
    clf,
    x_data,
    y_target,
    target_name="progress",
    feature_names=x_data.columns,
    class_names=["low", "mid", "high"],
    fontname="IPAGothic",
    # 試しに"high"のデータを１つ追ってみる
    X=x_data.iloc[0, :],
)

viz

#%%
low_df = quest_df[quest_df["progress_name"] == "low"].reset_index(drop=True)
mid_df = quest_df[quest_df["progress_name"] == "mid"].reset_index(drop=True)
high_df = quest_df[quest_df["progress_name"] == "high"].reset_index(drop=True)


for df in [low_df, mid_df, high_df]:
    quest_list = []
    for i in df.columns[:28]:
        completed_quests = {}
        completed_quests["id"] = i
        completed_quests["count"] = len(df[df[i] == True])
        quest_list.append(completed_quests)
    temp_df = pd.DataFrame(quest_list)
    name = str(df.loc[0, "progress_name"])
    temp_df.rename(columns={"count": name}, inplace=True)
    quest_completed_df = quest_completed_df.merge(temp_df, on="id")
quest_completed_df
#%%


#%%
x = quest_completed_df["description"]
y1 = quest_completed_df["low"][::-1]
y2 = quest_completed_df["mid"][::-1]
y3 = quest_completed_df["high"][::-1]

fig = plt.figure(figsize=(10, 15))
ax = fig.add_subplot(1, 1, 1)
ax.barh(x, y1, height=0.7, tick_label=x[::-1], label="low")
ax.barh(x, y2, height=0.7, tick_label=x[::-1], label="mid")
ax.barh(x, y3, height=0.7, tick_label=x[::-1], label="high")
ax.legend(fontsize=10)

#%%
progress_df = quest_completed_df[["low", "mid", "high"]].T
fig = plt.figure(figsize=(10, 15))
ax = fig.add_subplot(1, 1, 1)
x = quest_completed_df["description"]
for i in range(len(progress_df)):
    ax.bar(x, progress_df.iloc[i], bottom=progress_df.iloc[:i].sum())
fig.autofmt_xdate(rotation=90)
ax.legend(progress_df.index, fontsize=16)
plt.show()


# %%
sns.pairplot(badge_df)
# %%
len(result)
# %%
badge_df = pd.DataFrame(user_info_list)
badge_df.to_json("user_info.json")
# %%
# %%
# %%
import statsmodels.api as sm

x_add_const = sm.add_constant(x)
model_sm = sm.OLS(y, x_add_const).fit()
print(model_sm.summary())
# %%
# 単相関
corr = badge_df["cleard_quests"].corr(badge_df["player_xp"])
print(corr)
# %%
# 散布図
plt.scatter(badge_df["cleard_quests"], badge_df["player_xp"])
plt.yscale("log")
plt.yticks([10 ** i for i in range(7)])
plt.show()
# %%
# 総相関
mask = np.zeros_like(badge_df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.subplots(figsize=(12, 12))
sns.set_context("talk")
ax = sns.heatmap(
    badge_df.corr(),
    vmin=-1,
    vmax=1,
    mask=mask,
    square=True,
    annot=True,
    linewidths=0.5,
    fmt="g",
)
plt.show()
# %%
# 単回帰分析
X = badge_df[["cleard_quests"]]
Y = badge_df[["player_xp"]]
LinerRegr = linear_model.LinearRegression()
LinerRegr.fit(X, Y)
print("- R^2: %s" % LinerRegr.score(X, Y))
print(
    "- 回帰式: y = %s*x %s"
    % (round(float(LinerRegr.coef_), 5), round(float(LinerRegr.intercept_), 5))
)
px = np.arange(int(X.min()), int(X.max()), 0.1)[:, np.newaxis]
py = LinerRegr.predict(px)
plt.subplots(figsize=(7, 6))
plt.plot(px, py, color="blue", linewidth=3)
plt.scatter(X, Y, color="black")
plt.xlabel(X.columns[0])
plt.ylabel(Y.columns[0])
plt.yscale("log")
plt.yticks([10 ** i for i in range(7)])
plt.show()
# %%size
# %%
result = sm.ols(formula="np.log(cleard_quests) ~ player_xp", data=badge_df).fit()

# 概要だけ
result.params
#%%
# 分析データ含めて表示
result.summary()
# %%
# statsmodelsで単回帰分析
# 最小二乗法(ols)を使った回帰
# formula = 従属変数 ~ 説明変数
result = sm.ols(formula="np.log(player_xp) ~ cleard_quests", data=badge_df).fit()
# %%
min(badge_df["player_xp"])
# %%
