"""
Lesson 03: スコープ — user_id / agent_id / run_id
=================================================

mem0 の記憶は 3 種類の ID で「誰の・どの」記憶かを区別します。

  user_id   ユーザー単位の長期記憶      例: 好み、属性、目標        （何か月も残る）
  agent_id  エージェント単位の記憶       例: エージェント自身の方針  （ユーザー横断）
  run_id    セッション単位の短期記憶     例: 今回の旅行計画の条件     （終わったら消す）

組み合わせも自由です（例: user_id + run_id = あるユーザーのある会話セッション）。
検索時は filters に指定した ID すべてに一致するものだけが返ります。

実行:  uv run python lessons/03_scopes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tutorial.util import fresh_memory, note, section, show  # noqa: E402

m = fresh_memory("lesson03")

# ---------------------------------------------------------------------------
section("記憶を 3 つのスコープに保存")
# 長期: ユーザーのプロフィール
m.add("ユーザーは乗り物酔いしやすい", user_id="yamada", infer=False)
m.add("ユーザーは温泉が好き", user_id="yamada", infer=False)
# 短期: この旅行計画セッションだけの条件
m.add("今回の旅行の予算は5万円", user_id="yamada", run_id="trip-2026-12", infer=False)
m.add("今回の旅行は12月の2泊3日", user_id="yamada", run_id="trip-2026-12", infer=False)
# エージェント: 旅行エージェント自身の運用ルール（全ユーザー共通）
m.add("旅程を提案するときは移動時間を必ず併記する", agent_id="travel-agent", infer=False)
note("ここでは結果を分かりやすくするため infer=False で直接書き込んでいます。")

# ---------------------------------------------------------------------------
section("filters={'user_id': 'yamada'} → ユーザーの全記憶（セッション分も含む）")
show(m.get_all(filters={"user_id": "yamada"}), keys=("memory", "run_id"))

section("filters={'user_id': 'yamada', 'run_id': 'trip-2026-12'} → このセッションだけ")
show(m.get_all(filters={"user_id": "yamada", "run_id": "trip-2026-12"}), keys=("memory", "run_id"))

section("filters={'agent_id': 'travel-agent'} → エージェント固有のノウハウ")
show(m.get_all(filters={"agent_id": "travel-agent"}), keys=("memory", "agent_id"))

# ---------------------------------------------------------------------------
section("セッション終了: run_id の記憶だけを消す")
show(m.delete_all(run_id="trip-2026-12"))
show(m.get_all(filters={"user_id": "yamada"}), keys=("memory", "run_id"))
note("長期記憶（乗り物酔い・温泉好き）は残り、今回の旅行条件だけが消えました。")

# ---------------------------------------------------------------------------
section("よくある間違い")
try:
    m.search("温泉", user_id="yamada")  # 2.x ではエラー
except Exception as e:
    print(f"  {type(e).__name__}: {e}")
note("search()/get_all() は filters={'user_id': ...} で指定します。add() だけはトップレベル引数です。")

try:
    m.get_all()
except Exception as e:
    print(f"  {type(e).__name__}: {e}")
note("スコープ無しの全件取得はできません（他ユーザーの記憶の漏洩を防ぐため）。")
