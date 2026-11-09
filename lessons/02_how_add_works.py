"""
Lesson 02: add() の中身 — infer / 重複排除 / カスタム指示
=========================================================

学ぶこと
  - infer=True（既定）と infer=False の違い
  - 同じ事実を 2 回入れたときの重複排除
  - 事実が「変わった」ときの挙動（2.x は自動で上書きしない）
  - custom_instructions / prompt で「何を覚えるか」をコントロールする

add(infer=True) の内部処理（mem0 2.x）:
  Phase 1  既存の記憶から関連しそうなもの上位10件を検索
  Phase 2  LLM 1 回で「新しく覚えるべき事実」を JSON で抽出（既存記憶・直近の会話も参考にする）
  Phase 3  抽出した事実をまとめて埋め込み
  Phase 4  MD5 ハッシュで完全一致の重複を除外
  Phase 5  ベクトルストアへ保存 + history.db に ADD を記録
  Phase 6  固有名詞などのエンティティを抽出してリンク（検索時のブーストに使う）

実行:  uv run python lessons/02_how_add_works.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tutorial.util import fresh_memory, note, section, show  # noqa: E402

m = fresh_memory("lesson02")
U = {"user_id": "sato"}

# ---------------------------------------------------------------------------
section("infer=True: LLM が事実を抽出する（既定）")
show(m.add("えーと、実は来月からシアトルに転勤になるんですよ。英語は苦手なので不安です…", **U),
     keys=("memory", "event"))

section("infer=False: 文字列をそのまま 1 件の記憶として保存（LLM 呼び出し無し）")
show(m.add("ユーザーのTシャツのサイズはL", **U, infer=False), keys=("memory", "event"))
note("infer=False は高速・無料。アプリ側で既に整形済みの事実（フォーム入力など）を入れるときに便利です。")
note("エージェントが自分で『これを覚える』と決めるツール型メモリ（Lesson 07）でもこちらを使います。")

# ---------------------------------------------------------------------------
section("重複: 同じ内容をもう一度 add する")
print("  infer=False でもう一度:")
show(m.add("ユーザーのTシャツのサイズはL", **U, infer=False), keys=("memory", "event"))
note("infer=False は重複チェックをしません（そのまま 2 件目が入る）。重複を避けたいならアプリ側で search してから入れます。")
print("  infer=True で言い換えて:")
show(m.add("来月シアトルに転勤します", **U), keys=("memory", "event"))
note("infer=True では、関連する既存記憶が LLM に渡され『既知』の事実は抽出されません。")
note("さらに抽出結果が既存記憶と完全一致した場合は MD5 ハッシュで除外されます。")

# ---------------------------------------------------------------------------
section("事実が変わったとき")
show(m.add("やっぱり転勤はなくなって、東京に残ることになりました", **U), keys=("memory", "event"))
show(m.get_all(filters=U), keys=("memory", "created_at"))
note("mem0 2.x の add は既存記憶を自動で UPDATE/DELETE しません（追加専用・監査しやすい設計）。")
note("古い事実も残るので、エージェント側で『新しい日付を優先』と指示するか、update()/delete() で整理します。")
note("→ 本チュートリアルのエージェントは、記憶に記録日を付けてプロンプトに入れています（tutorial/agent.py）。")

# ---------------------------------------------------------------------------
section("custom_instructions: 何を覚えるかを指示する")
m2 = fresh_memory(
    "lesson02_custom",
    custom_instructions=(
        "料理アプリ用のメモリです。食の好み・アレルギー・食事制限・調理環境だけを記録してください。"
        "仕事や家族など、食に関係ない情報は記録しないでください。"
    ),
)
show(
    m2.add(
        "私はエビアレルギーで、在宅勤務のプログラマーです。家にはオーブンがなくて電子レンジだけです。",
        user_id="sato",
    ),
    keys=("memory",),
)
note("用途に関係ない個人情報を溜め込まないのは、品質面でもプライバシー面でも重要です。")
note("add(..., prompt='...') で、その呼び出しだけ指示を変えることもできます。")
