"""
Lesson 04: 検索を深掘り — スコアの内訳 / threshold / メタデータフィルタ / 有効期限
==============================================================================

search() のスコアは 3 つのシグナルの組み合わせです（mem0 2.x）。

  semantic_score  埋め込みベクトルのコサイン類似度（意味の近さ）
  bm25_score      キーワード一致度（BM25 を 0〜1 に正規化）
  entity_boost    質問と記憶に同じ固有名詞などが含まれると加点

  explain=True を付けると、各結果に score_details として内訳が付きます。

実行:  uv run python lessons/04_search_and_filters.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tutorial.util import fresh_memory, note, section, show  # noqa: E402

m = fresh_memory("lesson04")
U = "kato"

# metadata には任意の情報を付けられ、検索時のフィルタに使えます
facts = [
    ("ユーザーはPythonとTypeScriptが得意", {"category": "skill", "importance": 3}),
    ("ユーザーはRustを勉強中", {"category": "skill", "importance": 2}),
    ("ユーザーはコーヒーよりも紅茶派", {"category": "preference", "importance": 1}),
    ("ユーザーは毎朝6時にジョギングする", {"category": "habit", "importance": 2}),
    ("ユーザーは来週の金曜に歯医者の予約がある", {"category": "schedule", "importance": 3}),
]
for text, meta in facts:
    m.add(text, user_id=U, metadata=meta, infer=False)

# 有効期限付きの記憶（期限を過ぎると検索に出なくなる）
m.add("ユーザーは期間限定セールのクーポンを持っている", user_id=U, infer=False,
      metadata={"category": "schedule"}, expiration_date="2020-01-01")

# ---------------------------------------------------------------------------
section("explain=True でスコアの内訳を見る")
res = m.search("プログラミング言語の経験", filters={"user_id": U}, top_k=3, explain=True)
for r in res["results"]:
    d = r["score_details"]
    print(f"  {r['memory']}\n     final={d['final_score']:.3f}  semantic={d['semantic_score']:.3f}"
          f"  bm25={d['bm25_score']:.3f}  entity={d['entity_boost']:.3f}")
note("BM25 の見出し語化とエンティティ抽出は spaCy の英語モデルを使います（pip install 'mem0ai[nlp]'）。")
note("日本語の記憶では主に semantic_score が効く、と考えておくと挙動を理解しやすいです。")

section("threshold: スコアの足切り（既定 0.1）")
for th in (0.0, 0.3):
    res = m.search("飲み物", filters={"user_id": U}, threshold=th)
    print(f"  threshold={th}: {len(res['results'])} 件 → {[r['memory'] for r in res['results']]}")
note("小さめの埋め込みモデル（fastembed の MiniLM など）はスコアが低めに出ます。件数が少なすぎたら下げましょう。")
note("逆に、無関係な記憶がプロンプトに混ざるなら上げます。top_k と合わせてチューニングします。")

# ---------------------------------------------------------------------------
section("メタデータフィルタ: 完全一致")
show(m.get_all(filters={"user_id": U, "category": "skill"}), keys=("memory", "metadata"))

section("メタデータフィルタ: 演算子（gte / in / NOT など）")
show(m.search("ユーザーについて", filters={"user_id": U, "importance": {"gte": 3}}), keys=("memory", "metadata"))
show(m.search("ユーザーについて", filters={"user_id": U, "category": {"in": ["habit", "preference"]}}),
     keys=("memory", "metadata"))
note("使える演算子: eq, ne, in, nin, gt, gte, lt, lte, contains, icontains, AND, OR, NOT")
note("ベクトルストアによって対応状況が異なる場合があります（ここではローカル Qdrant）。")

# ---------------------------------------------------------------------------
section("expiration_date: 期限切れの記憶")
print("  通常:", [r["memory"] for r in m.get_all(filters={"user_id": U, "category": "schedule"})["results"]])
print("  show_expired=True:",
      [r["memory"] for r in m.get_all(filters={"user_id": U, "category": "schedule"}, show_expired=True)["results"]])
note("『今週の予定』のような賞味期限のある情報は expiration_date を付けておくと自動で検索対象外になります。")
