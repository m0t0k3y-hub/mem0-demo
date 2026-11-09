"""
Lesson 08: マルチエージェントでメモリを共有する（パターン C）
===========================================================

専門分野の違う複数のエージェントが、同じユーザーの記憶を共有するパターンです。

                    ┌──────────────────────────────┐
                    │ user_id="watanabe" の記憶     │ ← どのエージェントも読み書きする
                    │ 「卵アレルギー」「京都旅行予定」│   （= エージェント間の引き継ぎ）
                    └──────┬───────────────┬───────┘
                           │               │
              ┌────────────▼───┐     ┌─────▼──────────┐
              │ 旅行プランナー   │     │ 料理アドバイザー │
              └────────────┬───┘     └─────┬──────────┘
                           │               │
         agent_id="travel" の記憶   agent_id="food" の記憶
         （旅行エージェントのノウハウ） （料理エージェントのノウハウ）

  - ユーザーの事実は user_id だけで保存 → 全エージェントで共有
    （metadata.source_agent で、どのエージェントが聞き出したかを残しておくと監査しやすい）
  - エージェント固有の知識は agent_id だけで保存 → 全ユーザーに使い回す

実行:  uv run python lessons/08_multi_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tutorial.agent import MemoryChatAgent  # noqa: E402
from tutorial.llm import ChatLLM  # noqa: E402
from tutorial.util import fresh_memory, note, section, show  # noqa: E402

m = fresh_memory("lesson08")
llm = ChatLLM()  # LLM クライアントは共有して OK
USER = "watanabe"

travel = MemoryChatAgent(
    m, llm, agent_id="travel",
    persona="あなたは旅行プランナーです。ユーザーの記憶を踏まえて、具体的で簡潔な旅行提案をしてください。",
)
food = MemoryChatAgent(
    m, llm, agent_id="food",
    persona="あなたは料理・グルメのアドバイザーです。ユーザーの記憶を踏まえて、安全で具体的な提案をしてください。",
)

# ---------------------------------------------------------------------------
section("各エージェントのノウハウ（agent_id スコープ）を登録")
m.add("旅程を提案するときは移動時間の目安を必ず書く", agent_id="travel", infer=False)
m.add("飲食店を勧めるときはアレルギー対応の可否を確認するよう促す", agent_id="food", infer=False)
note("運用者が登録する『手順・方針』の記憶。プロンプトに直書きする代わりに、関連する場面だけ取り出せます。")


def ask(agent: MemoryChatAgent, name: str, message: str) -> None:
    turn = agent.chat(USER, message)
    print(f"\n👤 → {name}: {message}")
    print("   🔍 " + (" / ".join(r["memory"] for r in turn.retrieved) or "（記憶なし）"))
    print(f"🤖 {name}: {turn.reply}")
    if turn.stored:
        print("   💾 " + " / ".join(s["memory"] for s in turn.stored))


# ---------------------------------------------------------------------------
section("旅行プランナーに相談（ここでアレルギーの話が出る）")
ask(travel, "旅行", "12月に京都へ1泊2日で行きたいです。卵アレルギーがあるので食事が少し心配です。")

section("料理アドバイザーに相談（アレルギーの話はしていない）")
ask(food, "料理", "京都で夕食を食べるならどんなお店がいい？")
note("料理アドバイザーには一度も伝えていないのに、卵アレルギーと京都旅行を踏まえて答えられます。")

# ---------------------------------------------------------------------------
section("記憶の中身: 誰が聞き出した事実か")
show(m.get_all(filters={"user_id": USER}), keys=("memory", "metadata"))
section("agent_id スコープの記憶（ユーザーの事実とは混ざらない）")
show(m.get_all(filters={"agent_id": "food"}), keys=("memory", "agent_id"))
