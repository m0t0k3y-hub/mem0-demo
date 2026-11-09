"""
Lesson 06: エージェント × メモリ（パターン A: 自動メモリ）
=========================================================

いちばん基本で、いちばんよく使われる組み合わせ方です。

    ユーザー発話
        │
        ▼
  ① Retrieve   memory.search(発話, filters={"user_id": ...})
        │        └─ 関連する記憶 top_k 件
        ▼
  ② Generate   システムプロンプトに記憶を差し込んで LLM が応答
        │
        ▼
  ③ Store      memory.add([user発話, assistant応答], user_id=...)
                 └─ LLM が「覚えるべき事実」だけを抽出して保存

ポイント: 会話履歴（短期記憶）は数ターン分だけ LLM に渡し、
          それより古い内容や別セッションの内容は mem0（長期記憶）から思い出させる。

このレッスンでは「セッション 1 で話した内容を、履歴を捨てたセッション 2 で覚えているか」を確かめます。
エージェントの実装は tutorial/agent.py の MemoryChatAgent を見てください。

実行:  uv run python lessons/06_agent_auto_memory.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tutorial.agent import MemoryChatAgent  # noqa: E402
from tutorial.util import fresh_memory, note, section  # noqa: E402

m = fresh_memory("lesson06")
agent = MemoryChatAgent(m)
USER = "nakamura"


def talk(message: str, history: list[dict]) -> None:
    turn = agent.chat(USER, message, history)
    print(f"\n👤 {message}")
    if turn.retrieved:
        print("   🔍 思い出した記憶: " + " / ".join(r["memory"] for r in turn.retrieved))
    print(f"🤖 {turn.reply}")
    if turn.stored:
        print("   💾 新しく覚えた: " + " / ".join(s["memory"] for s in turn.stored))
    t = turn.timings
    print(f"   ⏱ retrieve {t['retrieve']:.1f}s / generate {t['generate']:.1f}s / store {t['store']:.1f}s")
    history += [{"role": "user", "content": message}, {"role": "assistant", "content": turn.reply}]


# ---------------------------------------------------------------------------
section("セッション 1: 自己紹介")
history: list[dict] = []
talk("はじめまして。中村です。来年フルマラソンに初挑戦する予定で、今は週3回走っています。", history)
talk("膝を痛めやすいのが悩みです。あと、朝は時間がないので夜に練習しています。", history)

# ---------------------------------------------------------------------------
section("セッション 2: 会話履歴をリセット（= 数日後に新しいチャットを開いた想定）")
history = []
talk("今週のトレーニングメニューを考えてくれる？", history)
note("会話履歴は空なのに、膝の不安・夜練習・マラソン目標を踏まえた提案になっているはずです。")

# ---------------------------------------------------------------------------
section("比較: メモリ無しのエージェント")
plain = MemoryChatAgent(m, use_memory=False)
print("🤖", plain.chat(USER, "今週のトレーニングメニューを考えてくれる？").reply)
note("一般論の回答になり、ユーザーに条件を聞き返すことが多くなります。")

section("この時点の記憶一覧")
for r in m.get_all(filters={"user_id": USER})["results"]:
    print("  -", r["memory"])
