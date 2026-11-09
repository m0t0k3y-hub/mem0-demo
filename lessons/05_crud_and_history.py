"""
Lesson 05: 記憶の管理 — get / update / history / delete / delete_all / reset
===========================================================================

本番のアプリでは「ユーザーが自分の記憶を確認・修正・削除できる」ことが重要です
（誤った記憶の訂正、プライバシー対応、GDPR の『忘れられる権利』など）。

実行:  uv run python lessons/05_crud_and_history.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tutorial.util import fresh_memory, note, section, show  # noqa: E402

m = fresh_memory("lesson05")
U = {"user_id": "ito"}

m.add("ユーザーは名古屋に住んでいる", **U, infer=False)
m.add("ユーザーはサッカー観戦が趣味", **U, infer=False)
m.add("ユーザーの誕生日は3月3日", **U, infer=False)

memories = m.get_all(filters=U)["results"]
target = next(x for x in memories if "名古屋" in x["memory"])
mid = target["id"]

section("get(): ID を指定して 1 件取得")
show({k: v for k, v in m.get(mid).items() if k not in ("hash", "text_lemmatized")})

section("update(): 内容を書き換える")
show(m.update(mid, "ユーザーは福岡に住んでいる（2026年11月に名古屋から引っ越し）"))
print("  更新後:", m.get(mid)["memory"])

section("history(): 変更履歴（history.db に記録されている）")
for h in m.history(mid):
    print(f"  {h['event']:<6} {h['old_memory']!s:<20} → {h['new_memory']}")
note("いつ・何が・どう変わったかを追跡できるので、誤った更新の調査や巻き戻しに使えます。")

section("delete(): 1 件削除")
bday = next(x for x in memories if "誕生日" in x["memory"])
show(m.delete(bday["id"]))
show(m.get_all(filters=U), keys=("memory",))
for h in m.history(bday["id"]):
    print(f"  history: {h['event']} {h['old_memory']} → {h['new_memory']} (is_deleted={h['is_deleted']})")

section("delete_all(): スコープ単位で一括削除（例: 退会したユーザー）")
show(m.delete_all(user_id="ito"))
show(m.get_all(filters=U))

section("reset(): すべての記憶と履歴を消去（開発・テスト用）")
m.add("テスト用の記憶", user_id="tmp", infer=False)
m.reset()
show(m.get_all(filters={"user_id": "tmp"}))
note("reset() は全ユーザー分が消えます。本番では delete_all(user_id=...) を使いましょう。")
