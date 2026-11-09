"""
mem0 チュートリアル — Gradio UI
==============================

起動:  uv run python app.py   →  http://127.0.0.1:7860

タブ構成
  1. はじめに          … mem0 の全体像と学習ロードマップ
  2. API プレイグラウンド … add / search / get_all / update / delete / history を GUI で試す
                         （実行するたびに、同じ処理の Python コードを表示）
  3. エージェント        … パターン A（自動メモリ）/ B（ツール型）を対話で比較
  4. マルチエージェント   … 旅行・料理エージェントが記憶を共有する様子を見る
  5. レッスン            … lessons/*.py のコードを読み、その場で実行して出力を見る
  6. 設計のコツ          … 本番で使うときのベストプラクティス
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import gradio as gr

from tutorial import util  # noqa: F401  (警告・ログの抑制)
from tutorial.agent import MemoryChatAgent, ToolMemoryAgent
from tutorial.config import PROJECT_ROOT, build_memory, load_settings
from tutorial.llm import ChatLLM

LESSON_DIR = PROJECT_ROOT / "lessons"
DOCS_DIR = PROJECT_ROOT / "docs"
HISTORY_TURNS = 6  # LLM に渡す直近の会話ターン数（短期記憶）

# ---------------------------------------------------------------------------
# Memory / LLM は最初に使うときに作る（設定ミスがあっても UI は起動させるため）
# ---------------------------------------------------------------------------
_state: dict = {}


def mem():
    if "memory" not in _state:
        _state["memory"] = build_memory("app")
    return _state["memory"]


def llm() -> ChatLLM:
    if "llm" not in _state:
        _state["llm"] = ChatLLM()
    return _state["llm"]


def settings_text() -> str:
    try:
        return "✅ " + load_settings().describe()
    except Exception as e:
        return f"⚠️ 設定エラー: {e}"


def err(e: Exception) -> str:
    return f"❌ {type(e).__name__}: {e}\n\n{traceback.format_exc(limit=3)}"


def jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def strip(items: list[dict]) -> list[dict]:
    hidden = {"hash", "text_lemmatized"}
    return [{k: v for k, v in it.items() if k not in hidden and v is not None} for it in items]


def py(value) -> str:
    """Python コード表示用のリテラル"""
    return repr(value)


def parse_json(text: str, name: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        v = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{name} が JSON として読めません: {e}") from e
    if not isinstance(v, dict):
        raise ValueError(f"{name} は JSON オブジェクト（{{...}}）で指定してください")
    return v


def parse_messages(text: str):
    """'user: ...' / 'assistant: ...' 形式の行を messages に変換。接頭辞が無ければ文字列のまま。"""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        raise ValueError("メッセージを入力してください")
    if not any(ln.lower().startswith(("user:", "assistant:")) for ln in lines):
        return text.strip()
    msgs = []
    for ln in lines:
        role, _, content = ln.partition(":")
        role = role.strip().lower()
        if role in ("user", "assistant"):
            msgs.append({"role": role, "content": content.strip()})
        elif msgs:
            msgs[-1]["content"] += "\n" + ln.strip()
    return msgs


# =========================================================================== 2. API Playground
TABLE_COLS = ["id", "memory", "user_id", "agent_id", "run_id", "metadata", "updated_at"]


def list_memories(filters_text: str):
    try:
        filters = parse_json(filters_text, "filters") or {}
        res = mem().get_all(filters=filters, top_k=100)["results"]
        rows = [[r.get("id"), r.get("memory"), r.get("user_id"), r.get("agent_id"), r.get("run_id"),
                 json.dumps(r.get("metadata"), ensure_ascii=False) if r.get("metadata") else "",
                 (r.get("updated_at") or "")[:19]] for r in res]
        code = f"m.get_all(filters={py(filters)}, top_k=100)"
        return rows, f"{len(rows)} 件", code
    except Exception as e:
        return [], err(e), ""


def do_add(text, user_id, agent_id, run_id, infer, metadata_text, expiration):
    try:
        messages = parse_messages(text)
        kwargs: dict = {}
        for k, v in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)):
            if v and v.strip():
                kwargs[k] = v.strip()
        metadata = parse_json(metadata_text, "metadata")
        if metadata:
            kwargs["metadata"] = metadata
        if expiration and expiration.strip():
            kwargs["expiration_date"] = expiration.strip()
        kwargs["infer"] = bool(infer)

        code = "messages = " + jdump(messages) + "\n\nresult = m.add(\n    messages,\n" + "".join(
            f"    {k}={py(v)},\n" for k, v in kwargs.items()) + ")"
        result = mem().add(messages, **kwargs)
        out = strip(result.get("results", []))
        msg = f"{len(out)} 件の記憶が追加されました" if out else "追加なし（覚える事実が無い／重複と判定）"
        return jdump(out), msg, code
    except Exception as e:
        return "", err(e), ""


def do_search(query, filters_text, top_k, threshold, explain):
    try:
        filters = parse_json(filters_text, "filters") or {}
        code = (f"results = m.search(\n    {py(query)},\n    filters={py(filters)},\n"
                f"    top_k={int(top_k)},\n    threshold={threshold},\n    explain={bool(explain)},\n)")
        res = mem().search(query, filters=filters, top_k=int(top_k), threshold=float(threshold),
                           explain=bool(explain))["results"]
        return jdump(strip(res)), f"{len(res)} 件ヒット", code
    except Exception as e:
        return "", err(e), ""


def do_get(memory_id):
    try:
        return jdump(strip([mem().get(memory_id.strip())])[0]), f"m.get({py(memory_id.strip())})"
    except Exception as e:
        return err(e), ""


def do_update(memory_id, new_text):
    try:
        res = mem().update(memory_id.strip(), new_text)
        return jdump(res), f"m.update({py(memory_id.strip())}, {py(new_text)})"
    except Exception as e:
        return err(e), ""


def do_history(memory_id):
    try:
        return jdump(mem().history(memory_id.strip())), f"m.history({py(memory_id.strip())})"
    except Exception as e:
        return err(e), ""


def do_delete(memory_id):
    try:
        return jdump(mem().delete(memory_id.strip())), f"m.delete({py(memory_id.strip())})"
    except Exception as e:
        return err(e), ""


def do_delete_all(filters_text):
    try:
        f = parse_json(filters_text, "filters") or {}
        scope = {k: v for k, v in f.items() if k in ("user_id", "agent_id", "run_id")}
        if not scope:
            raise ValueError("filters に user_id / agent_id / run_id のいずれかが必要です")
        args = ", ".join(f"{k}={py(v)}" for k, v in scope.items())
        return jdump(mem().delete_all(**scope)), f"m.delete_all({args})"
    except Exception as e:
        return err(e), ""


def on_select_row(evt: gr.SelectData):
    row = getattr(evt, "row_value", None)
    return row[0] if row else gr.update()


# =========================================================================== 3. Agent
def _memory_md(items: list[dict], empty: str) -> str:
    if not items:
        return empty
    lines = []
    for r in items:
        score = f" `{r['score']:.2f}`" if isinstance(r.get("score"), (int, float)) else ""
        ev = f"**{r['event']}** " if r.get("event") else ""
        lines.append(f"- {ev}{r.get('memory')}{score}")
    return "\n".join(lines)


def _user_memories_md(user_id: str) -> str:
    try:
        res = mem().get_all(filters={"user_id": user_id}, top_k=100)["results"]
    except Exception as e:
        return f"❌ {e}"
    if not res:
        return "（まだ記憶はありません）"
    return "\n".join(
        f"- {r['memory']}" + (f"  _({r['metadata'].get('source_agent')})_"
                              if (r.get("metadata") or {}).get("source_agent") else "")
        for r in res
    )


def agent_chat(message, chat, llm_history, pattern, user_id, use_memory, top_k, threshold):
    if not message.strip():
        return gr.update(), chat, llm_history, gr.update(), gr.update(), gr.update(), gr.update()
    user_id = user_id.strip() or "demo-user"
    try:
        if pattern.startswith("A"):
            agent = MemoryChatAgent(mem(), llm(), use_memory=use_memory, top_k=int(top_k),
                                    threshold=float(threshold))
        else:
            agent = ToolMemoryAgent(mem(), llm())
        turn = agent.chat(user_id, message, llm_history[-HISTORY_TURNS * 2:])
        reply = turn.reply
        retrieved = _memory_md(turn.retrieved, "（なし）")
        stored = _memory_md(turn.stored, "（なし）")
        detail = jdump({"tool_calls": turn.tool_events, "timings_sec": {k: round(v, 2) for k, v in turn.timings.items()}})
    except Exception as e:
        reply, retrieved, stored, detail = err(e), "", "", ""
    chat = chat + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    llm_history = llm_history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    return "", chat, llm_history, retrieved, stored, detail, _user_memories_md(user_id)


# =========================================================================== 4. Multi-agent
AGENTS = {
    "🧳 旅行プランナー": ("travel", "あなたは旅行プランナーです。ユーザーの記憶を踏まえて、具体的で簡潔な旅行提案をしてください。"),
    "🍳 料理アドバイザー": ("food", "あなたは料理・グルメのアドバイザーです。ユーザーの記憶を踏まえて、安全で具体的な提案をしてください。"),
    "💪 フィットネスコーチ": ("fitness", "あなたはフィットネスコーチです。ユーザーの記憶を踏まえて、無理のない運動を提案してください。"),
}
AGENT_KNOWHOW = {
    "travel": "旅程を提案するときは移動時間の目安を必ず書く",
    "food": "飲食店を勧めるときはアレルギー対応の可否を確認するよう促す",
    "fitness": "運動メニューには必ずウォームアップとクールダウンを含める",
}


def seed_knowhow():
    try:
        for aid, text in AGENT_KNOWHOW.items():
            if not mem().get_all(filters={"agent_id": aid})["results"]:
                mem().add(text, agent_id=aid, infer=False)
        return "\n".join(f"- `{a}`: {t}" for a, t in AGENT_KNOWHOW.items())
    except Exception as e:
        return f"❌ {e}"


def multi_chat(message, chat, agent_name, user_id):
    if not message.strip():
        return gr.update(), chat, gr.update(), gr.update()
    user_id = user_id.strip() or "demo-user"
    aid, persona = AGENTS[agent_name]
    try:
        agent = MemoryChatAgent(mem(), llm(), agent_id=aid, persona=persona)
        # エージェント同士は会話履歴を共有しない（共有するのは mem0 の記憶だけ）
        turn = agent.chat(user_id, message)
        reply = turn.reply
        info = (f"**{agent_name}** が思い出した記憶:\n{_memory_md(turn.retrieved, '（なし）')}\n\n"
                f"新しく覚えたこと:\n{_memory_md(turn.stored, '（なし）')}")
    except Exception as e:
        reply, info = err(e), ""
    chat = chat + [{"role": "user", "content": f"→ {agent_name}: {message}"},
                   {"role": "assistant", "content": f"{agent_name}: {reply}"}]
    return "", chat, info, _user_memories_md(user_id)


# =========================================================================== 5. Lessons
def lesson_files() -> list[str]:
    return sorted(p.name for p in LESSON_DIR.glob("[0-9]*.py"))


def load_lesson(name: str) -> str:
    return (LESSON_DIR / name).read_text(encoding="utf-8") if name else ""


def run_lesson(name: str):
    if not name:
        yield "レッスンを選んでください"
        return
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, str(LESSON_DIR / name)], cwd=PROJECT_ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = f"$ python lessons/{name}\n"
    noise = ("Exception ignored in", "sys.meta_path is None", "qdrant_client.py", "qdrant_local.py", "Traceback (most recent call last)")
    skip_next = False
    for line in proc.stdout:  # type: ignore[union-attr]
        if any(n in line for n in noise) or (skip_next and line.startswith("    ")):
            skip_next = True
            continue
        skip_next = False
        out += line
        yield out
    proc.wait()
    yield out + f"\n[終了コード {proc.returncode}]"


# =========================================================================== UI
def read_doc(name: str) -> str:
    return (DOCS_DIR / name).read_text(encoding="utf-8")


CSS = """
.small-note {font-size: 0.9em; opacity: 0.8}
"""

with gr.Blocks(title="mem0 チュートリアル") as demo:
    gr.Markdown("# 🧠 mem0 チュートリアル — メモリ × AI エージェント")
    gr.Markdown(settings_text(), elem_classes="small-note")

    # ------------------------------------------------------------------ 1
    with gr.Tab("1. はじめに"):
        gr.Markdown(read_doc("01_overview.md"))

    # ------------------------------------------------------------------ 2
    with gr.Tab("2. API プレイグラウンド"):
        gr.Markdown(
            "mem0 の API を GUI で直接呼び出します。右側に **実際に実行した Python コード** が表示されるので、"
            "そのままアプリに組み込めます（`m` は `Memory` インスタンス）。"
        )
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tab("add（記憶する）"):
                    add_text = gr.Textbox(
                        label="メッセージ（'user:' / 'assistant:' で始めると会話形式、無ければ文字列として渡す）",
                        lines=5,
                        value="user: 私は鈴木です。普段は福岡でデザイナーをしています。\n"
                              "assistant: 鈴木さん、よろしくお願いします！\n"
                              "user: 甘いものが苦手で、コーヒーはブラック派です。",
                    )
                    with gr.Row():
                        add_user = gr.Textbox(label="user_id", value="suzuki")
                        add_agent = gr.Textbox(label="agent_id", placeholder="任意")
                        add_run = gr.Textbox(label="run_id", placeholder="任意（セッションID）")
                    with gr.Row():
                        add_infer = gr.Checkbox(label="infer（LLM で事実を抽出）", value=True)
                        add_meta = gr.Textbox(label="metadata (JSON)", placeholder='{"category": "profile"}')
                        add_exp = gr.Textbox(label="expiration_date", placeholder="YYYY-MM-DD")
                    add_btn = gr.Button("m.add() を実行", variant="primary")
                    add_status = gr.Markdown()
                    add_out = gr.Code(label="戻り値 results", language="json")
                with gr.Tab("search（思い出す）"):
                    s_query = gr.Textbox(label="query", value="飲み物の好みは？")
                    s_filters = gr.Textbox(label="filters (JSON) — スコープ + メタデータ条件",
                                           value='{"user_id": "suzuki"}')
                    with gr.Row():
                        s_topk = gr.Slider(1, 20, value=5, step=1, label="top_k")
                        s_th = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="threshold")
                        s_explain = gr.Checkbox(label="explain（スコア内訳）", value=True)
                    s_btn = gr.Button("m.search() を実行", variant="primary")
                    s_status = gr.Markdown()
                    s_out = gr.Code(label="検索結果", language="json")
                with gr.Tab("管理（一覧・更新・削除・履歴）"):
                    with gr.Row():
                        l_filters = gr.Textbox(label="filters (JSON)", value='{"user_id": "suzuki"}', scale=4)
                        l_btn = gr.Button("m.get_all()", scale=1)
                    l_status = gr.Markdown()
                    l_table = gr.Dataframe(headers=TABLE_COLS, interactive=False, wrap=True,
                                           label="記憶一覧（行をクリックすると ID を選択）")
                    mid = gr.Textbox(label="memory_id")
                    new_text = gr.Textbox(label="update 用の新しい内容")
                    with gr.Row():
                        g_btn = gr.Button("get")
                        u_btn = gr.Button("update")
                        h_btn = gr.Button("history")
                        d_btn = gr.Button("delete", variant="stop")
                        da_btn = gr.Button("delete_all（filters のスコープ）", variant="stop")
                    m_out = gr.Code(label="結果", language="json")
            with gr.Column(scale=2):
                code_view = gr.Code(label="🐍 実行したコード", language="python", lines=14)
                gr.Markdown(read_doc("02_playground_tips.md"))

        add_btn.click(do_add, [add_text, add_user, add_agent, add_run, add_infer, add_meta, add_exp],
                      [add_out, add_status, code_view])
        s_btn.click(do_search, [s_query, s_filters, s_topk, s_th, s_explain], [s_out, s_status, code_view])
        l_btn.click(list_memories, [l_filters], [l_table, l_status, code_view])
        l_table.select(on_select_row, None, mid)
        g_btn.click(do_get, [mid], [m_out, code_view])
        u_btn.click(do_update, [mid, new_text], [m_out, code_view])
        h_btn.click(do_history, [mid], [m_out, code_view])
        d_btn.click(do_delete, [mid], [m_out, code_view])
        da_btn.click(do_delete_all, [l_filters], [m_out, code_view])

    # ------------------------------------------------------------------ 3
    with gr.Tab("3. エージェント"):
        gr.Markdown(read_doc("03_agent_patterns.md"))
        llm_hist = gr.State([])
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    pattern = gr.Radio(["A: 自動メモリ（Retrieve→Generate→Store）", "B: ツール型メモリ（LLM が判断）"],
                                       value="A: 自動メモリ（Retrieve→Generate→Store）", label="パターン")
                    a_user = gr.Textbox(label="user_id", value="demo-user")
                with gr.Row():
                    a_use = gr.Checkbox(label="メモリを使う（A のみ）", value=True)
                    a_topk = gr.Slider(1, 10, value=5, step=1, label="top_k（A のみ）")
                    a_th = gr.Slider(0.0, 0.8, value=0.1, step=0.05, label="threshold（A のみ）")
                chatbot = gr.Chatbot(height=420, label="会話")
                a_msg = gr.Textbox(placeholder="メッセージを入力して Enter", label="あなた")
                with gr.Row():
                    new_session = gr.Button("🔄 新しいセッション（会話履歴だけ消す・記憶は残る）")
                    gr.Examples(
                        [["はじめまして、佐々木です。札幌在住で、エビとカニのアレルギーがあります。"],
                         ["週末に家族で出かけたいんだけど、おすすめある？"],
                         ["今夜の夕飯の献立を考えて"],
                         ["実は先月、東京に引っ越しました"],
                         ["アレルギーの話は忘れてください"]],
                        inputs=[a_msg], label="例文",
                    )
            with gr.Column(scale=2):
                gr.Markdown("#### 🔍 思い出した記憶（このターン）")
                a_retrieved = gr.Markdown("—")
                gr.Markdown("#### 💾 保存・更新した記憶（このターン）")
                a_stored = gr.Markdown("—")
                with gr.Accordion("ツール呼び出し・処理時間", open=False):
                    a_detail = gr.Code(language="json")
                with gr.Accordion("このユーザーの全記憶", open=True):
                    a_all = gr.Markdown()
                    a_refresh = gr.Button("更新", size="sm")

        a_msg.submit(agent_chat, [a_msg, chatbot, llm_hist, pattern, a_user, a_use, a_topk, a_th],
                     [a_msg, chatbot, llm_hist, a_retrieved, a_stored, a_detail, a_all])
        new_session.click(lambda: ([], [], "", "", ""), None, [chatbot, llm_hist, a_retrieved, a_stored, a_detail])
        a_refresh.click(_user_memories_md, [a_user], [a_all])

    # ------------------------------------------------------------------ 4
    with gr.Tab("4. マルチエージェント"):
        gr.Markdown(read_doc("04_multi_agent.md"))
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    agent_pick = gr.Radio(list(AGENTS), value=list(AGENTS)[0], label="話しかけるエージェント")
                    m_user = gr.Textbox(label="user_id", value="multi-user")
                m_chat = gr.Chatbot(height=420, label="会話（エージェントごとに会話履歴は共有しない）")
                m_msg = gr.Textbox(placeholder="メッセージを入力して Enter", label="あなた")
                gr.Examples(
                    [["12月に京都へ1泊2日で行きたい。卵アレルギーがあるので食事が心配です。"],
                     ["京都で夕食を食べるならどんなお店がいい？"],
                     ["旅行中にできる軽い運動を教えて"]],
                    inputs=[m_msg], label="例文（上から順に、エージェントを切り替えながら送ってみてください）",
                )
            with gr.Column(scale=2):
                m_info = gr.Markdown()
                with gr.Accordion("エージェント固有のノウハウ（agent_id スコープ）", open=False):
                    m_knowhow = gr.Markdown()
                    seed_btn = gr.Button("ノウハウを登録", size="sm")
                with gr.Accordion("共有されているユーザーの記憶（user_id スコープ）", open=True):
                    m_all = gr.Markdown()
        m_msg.submit(multi_chat, [m_msg, m_chat, agent_pick, m_user], [m_msg, m_chat, m_info, m_all])
        seed_btn.click(seed_knowhow, None, m_knowhow)

    # ------------------------------------------------------------------ 5
    with gr.Tab("5. レッスン（コード＋実行）"):
        gr.Markdown("`lessons/` のスクリプトを読み、▶ で実行結果を確認できます。"
                    "各レッスンは専用のデータ領域（`data/lessonXX/`）を使い、毎回まっさらな状態から始まります。")
        with gr.Row():
            lesson_pick = gr.Dropdown(lesson_files(), value=lesson_files()[0], label="レッスン", scale=3)
            run_btn = gr.Button("▶ 実行", variant="primary", scale=1)
        with gr.Row():
            lesson_code = gr.Code(load_lesson(lesson_files()[0]), language="python", label="コード", lines=30)
            lesson_out = gr.Textbox(label="実行結果", lines=30, max_lines=60, autoscroll=True)
        lesson_pick.change(load_lesson, lesson_pick, lesson_code)
        # レッスンは別プロセス・別データ領域で動くので、他の操作と並行して実行してよい
        run_btn.click(run_lesson, lesson_pick, lesson_out, concurrency_limit=2, concurrency_id="lessons")

    # ------------------------------------------------------------------ 6
    with gr.Tab("6. 設計のコツ"):
        gr.Markdown(read_doc("05_best_practices.md"))


if __name__ == "__main__":
    # ローカル Qdrant はスレッドセーフではないので、イベントは 1 つずつ処理する
    demo.queue(default_concurrency_limit=1).launch(css=CSS)
