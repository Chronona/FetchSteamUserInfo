import os
from pathlib import Path

import pytest

# リポジトリ直下の steam_api.py を import できるようにする
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 動作確認に使う実在アカウント（St4ck 氏）
SAMPLE_UID = 76561198023414915


@pytest.fixture
def api_key():
    """スタブテスト用のダミーキー（実 API には送信されない）"""
    return "DUMMY_API_KEY"


@pytest.fixture(scope="session")
def real_api_key():
    """実 API 用のキー。環境変数 > .ignore/mydata/steam_api.txt の順に探す。

    見つからない場合はスモークテストを skip する。
    """
    key = os.environ.get("STEAM_API_KEY")
    if key:
        return key.strip()

    key_file = Path(__file__).resolve().parents[1] / ".ignore/mydata/steam_api.txt"
    if key_file.exists():
        return key_file.read_text().strip()

    pytest.skip(
        "API キーが無いため skip。環境変数 STEAM_API_KEY か "
        ".ignore/mydata/steam_api.txt を設定してください"
    )
