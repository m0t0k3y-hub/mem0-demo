"""
Lesson 01: クイックスタート — add / search / get_all
====================================================

学ぶこと
  - Memory の作り方（Memory() / Memory.from_config()）
  - add():    会話を渡すと、LLM が「覚えるべき事実」を抽出して保存する
  - search(): 質問に関係する記憶を意味検索する
  - get_all(): ユーザーの記憶を一覧する

実行:  uv run python lessons/01_quickstart.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tutorial.util import fresh_memory, note, section, show  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Memory を作る
#    最小構成なら `from mem0 import Memory; m = Memory()` だけで動きます
#    （OpenAI の LLM + 埋め込み、/tmp/qdrant のローカル DB を使う）。
#    ここでは .env の設定から組み立てる共通関数を使います（tutorial/config.py を参照）。
# ---------------------------------------------------------------------------
m = fresh_memory("lesson01")

# ---------------------------------------------------------------------------
# 2. add(): 会話を記憶する
#    messages は OpenAI 形式のメッセージのリスト（文字列 1 つでも可）。
#    user_id などの「スコープ」を最低 1 つ指定する必要があります。
# ---------------------------------------------------------------------------
section("add(): 会話から事実を抽出して保存")
conversation = [
    {"role": "user", "content": "こんにちは！私は田中です。大阪でWebエンジニアをしています。"},
    {"role": "assistant", "content": "田中さん、こんにちは！大阪でWebエンジニアをされているんですね。"},
    {"role": "user", "content": "辛い食べ物が大好きで、週末はよくカレー屋を巡っています。あと猫を2匹飼っています。"},
    {"role": "assistant", "content": "カレー巡り、いいですね！猫ちゃんも2匹いるんですね。"},
]
result = m.add(conversation, user_id="tanaka")
show(result, keys=("memory", "event"))
note("会話文そのものではなく、『ユーザーは大阪在住』のような短い事実に分解されて保存されます。")
note("mem0 2.x の add は『追加専用』: 1回の LLM 呼び出しで抽出 → 重複はハッシュで除外 → 保存。")

# ---------------------------------------------------------------------------
# 3. search(): 意味で検索する
#    スコープは filters={...} で指定します（mem0 2.x からの書き方）。
# ---------------------------------------------------------------------------
section("search(): 質問に関連する記憶を検索")
for query in ["今晩のご飯のおすすめは？", "ペットについて", "仕事は何をしている？"]:
    print(f"\n  Q: {query}")
    res = m.search(query, filters={"user_id": "tanaka"}, top_k=3)
    show(res, keys=("memory", "score"))
note("キーワードが一致しなくても、意味が近い記憶がヒットします（ベクトル検索 + BM25 のハイブリッド）。")

# ---------------------------------------------------------------------------
# 4. get_all(): スコープ内の記憶を全件取得
# ---------------------------------------------------------------------------
section("get_all(): ユーザーの記憶一覧")
show(m.get_all(filters={"user_id": "tanaka"}), keys=("id", "memory", "created_at"))

# 別ユーザーの記憶は混ざらない
section("スコープの分離: 別ユーザーで検索")
show(m.search("食べ物の好み", filters={"user_id": "suzuki"}))
note("user_id が違えば記憶は完全に分離されます。マルチテナントのアプリでも安全に使えます。")
