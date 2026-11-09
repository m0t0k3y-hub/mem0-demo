"""
Lesson 07: エージェント × メモリ（パターン B: メモリをツールとして渡す）
=====================================================================

パターン A ではアプリのコードが毎ターン必ず検索・保存しました。
パターン B では、記憶操作をツール（Function Calling）として LLM に渡し、
「いつ思い出すか」「何を覚えるか」「何を直すか」を LLM 自身に判断させます。

  search_memory(query)               → memory.search(...)
  save_memory(fact)                  → memory.add(fact, infer=False)  ※LLMが既に要約済み
  update_memory(memory_id, new_fact) → memory.update(...)
  delete_memory(memory_id)           → memory.delete(...)

            ┌──────────── LLM ────────────┐
  発話 ───▶ │ 「好みに関係する質問だ」     │──▶ search_memory("食の好み")
            │ 「新しい事実が出た」         │──▶ save_memory("ユーザーは…")
            │ 「引っ越したらしい」         │──▶ search → update_memory(id, …)
            └─────────────────────────────┘

パターン A との使い分け:
  A（自動）: 記憶漏れが許されないパーソナルアシスタント、サポート窓口など
  B（ツール）: 汎用エージェントで記憶が必要な場面が限られる／訂正・削除も会話でさせたい場合
  実務では「A で毎ターン軽く検索 + B で保存・訂正はツール」のハイブリッドも定番です。

実装は tutorial/agent.py の ToolMemoryAgent を見てください。

実行:  uv run python lessons/07_agent_memory_tools.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tutorial.agent import ToolMemoryAgent  # noqa: E402
from tutorial.util import fresh_memory, note, section  # noqa: E402

m = fresh_memory("lesson07")
agent = ToolMemoryAgent(m)
USER = "kobayashi"
history: list[dict] = []


def talk(message: str) -> None:
    turn = agent.chat(USER, message, history)
    print(f"\n👤 {message}")
    for e in turn.tool_events:
        args = json.dumps(e["args"], ensure_ascii=False)
        res = json.dumps(e["result"], ensure_ascii=False, default=str)
        print(f"   🛠 {e['tool']}({args})\n      → {res[:160]}")
    print(f"🤖 {turn.reply}")
    history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": turn.reply}])


section("会話の中で LLM が自分で記憶を操作する")
talk("こんにちは！小林です。横浜に住んでいて、休日は家族とキャンプに行くのが楽しみです。")
talk("今日はいい天気ですね。")  # 雑談 → 何も保存しないのが理想
talk("実は先月、横浜から札幌に引っ越したんです。")  # 既存の記憶を更新するはず
talk("今度の連休、どこに出かけるのがいいと思う？")  # 回答前に記憶を検索するはず
talk("あ、キャンプが趣味だっていう話は忘れてください。")  # 削除するはず

section("最終的な記憶")
for r in m.get_all(filters={"user_id": USER})["results"]:
    print("  -", r["memory"])
note("LLM の判断に任せる分、モデルの性能やシステムプロンプトの書き方で挙動が変わります。")
note("ツールの説明文（description）とシステムプロンプトの『いつ使うか』の記述が品質を左右します。")
