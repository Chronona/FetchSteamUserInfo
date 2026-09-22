# FetchSteamUserInfo

Steam Web API を使ってアカウント情報を取得する Jupyter Notebook。

解説記事: https://qiita.com/Chronona/items/2e464ee96799dd0ed5c2

## 構成

| パス | 内容 |
| --- | --- |
| `fetch_user_info.ipynb` | 調査の手順と結果 |
| `steam_api.py` | Steam Web API のラッパー |
| `tests/` | テスト（下記） |
| `docs/adr/` | 設計判断の記録 |

## できること

`fetch_user_info.ipynb` は次の流れを扱う。

1. `ISteamUser` / `IPlayerService` の各エンドポイントを叩いて 1 アカウントの情報を確認する
2. ランダムな SteamID を生成して実在するアカウントを探索する（`user_id.csv` に出力）
3. 見つかった ID の所持ゲーム・バッジ情報を取得する（`user_info.json` に出力）

## 実行手順

1. 依存パッケージをインストールする。

   ```sh
   pip install -r requirements.txt
   ```

2. [Steam Web API Key](https://steamcommunity.com/dev/apikey) を取得し、
   リポジトリ直下に次の 2 ファイルを置く（`.ignore/` は `.gitignore` 済み）。

   | パス | 内容 |
   | --- | --- |
   | `.ignore/mydata/steam_api.txt` | API キー |
   | `.ignore/mydata/account.txt` | 自分の SteamID（17 桁） |

3. `fetch_user_info.ipynb` を開いて上から実行する。
   「SteamID の収集」以降は数千〜1 万件のリクエストを行うため、実行に時間がかかる。

## テスト

```sh
pip install -r requirements-dev.txt
```

| コマンド | 内容 | API キー |
| --- | --- | --- |
| `pytest` | HTTP をスタブしたユニットテスト（1秒未満） | 不要 |
| `pytest -m smoke` | 実 API に 1 アカウント分だけアクセスし、レスポンスの仕様が変わっていないか確認 | 必要 |

スモークテストの API キーは環境変数 `STEAM_API_KEY`、または
`.ignore/mydata/steam_api.txt` から読み込む。どちらも無い場合は skip される。

テスト構成の意図は [ADR 0001](docs/adr/0001-api-wrapper-module-and-two-layer-tests.md) を参照。

## 参考

- Steam Web API ドキュメント: https://partner.steamgames.com/doc/webapi
- ISteamUser: https://partner.steamgames.com/doc/webapi/ISteamUser
- IPlayerService: https://partner.steamgames.com/doc/webapi/IPlayerService
