# mem0 チュートリアル — メモリ × AI エージェント

[mem0](https://github.com/mem0ai/mem0)（v2.2）の機能と、**長期記憶を AI エージェントにどう組み込むか** を、
動くサンプルコードと Gradio UI で学ぶためのチュートリアルです。

## セットアップ

```bash
uv sync
cp .env.example .env   # 使う LLM の API キーを 1 つ書く
```

| 使う LLM | `.env` に書くもの | 埋め込み（既定） |
|---|---|---|
| OpenAI | `OPENAI_API_KEY=sk-...` | OpenAI `text-embedding-3-small` |
| Anthropic (Claude) | `ANTHROPIC_API_KEY=sk-ant-...` | ローカルの fastembed（多言語 MiniLM。初回に約 200MB をダウンロード） |
| Ollama（ローカル・無料） | `LLM_PROVIDER=ollama`（事前に `ollama pull qwen3:8b`） | ローカルの fastembed |

ベクトル DB はローカルファイルの Qdrant（`data/` 以下）を使うため、DB サーバーは不要です。

## 起動

```bash
uv run python app.py
```

ブラウザで http://127.0.0.1:7860 を開きます。

レッスンは単体でも実行できます。

```bash
uv run python lessons/01_quickstart.py
```

## 学習の流れ

| # | 内容 | ファイル | LLM 呼び出し |
|---|---|---|---|
| 01 | クイックスタート: `add` / `search` / `get_all` | [lessons/01_quickstart.py](lessons/01_quickstart.py) | あり |
| 02 | `add` の中身: infer、重複、事実の変化、custom_instructions | [lessons/02_how_add_works.py](lessons/02_how_add_works.py) | あり |
| 03 | スコープ: `user_id` / `agent_id` / `run_id` | [lessons/03_scopes.py](lessons/03_scopes.py) | なし（埋め込みのみ） |
| 04 | 検索: スコアの内訳、threshold、メタデータフィルタ、有効期限 | [lessons/04_search_and_filters.py](lessons/04_search_and_filters.py) | なし（埋め込みのみ） |
| 05 | 管理: `get` / `update` / `history` / `delete` / `delete_all` / `reset` | [lessons/05_crud_and_history.py](lessons/05_crud_and_history.py) | なし（埋め込みのみ） |
| 06 | **エージェント パターン A**: 自動メモリ（Retrieve → Generate → Store） | [lessons/06_agent_auto_memory.py](lessons/06_agent_auto_memory.py) | あり |
| 07 | **エージェント パターン B**: メモリをツールとして LLM に渡す | [lessons/07_agent_memory_tools.py](lessons/07_agent_memory_tools.py) | あり |
| 08 | **パターン C**: マルチエージェントで記憶を共有 | [lessons/08_multi_agent.py](lessons/08_multi_agent.py) | あり |

Gradio UI のタブ:

1. **はじめに** — mem0 の全体像
2. **API プレイグラウンド** — 自分の文章で add / search / 更新 / 削除を試す。実行のたびに同じ処理の Python コードを表示
3. **エージェント** — パターン A/B、メモリ ON/OFF を切り替えて会話。思い出した記憶・保存した記憶・ツール呼び出しを横に表示
4. **マルチエージェント** — 旅行・料理・フィットネスのエージェントが記憶を共有する様子を見る
5. **レッスン** — `lessons/*.py` のコードを表示し、その場で実行
6. **設計のコツ** — 本番で使うためのベストプラクティス

## ファイル構成

```
app.py               Gradio UI
tutorial/
  config.py          .env → mem0 の設定（LLM / Embedder / Vector Store）
  llm.py             OpenAI・Ollama・Claude 共通のツール呼び出しループ
  agent.py           メモリ付きエージェント（パターン A / B / C）
  util.py            レッスン用の表示ヘルパー
lessons/             単体で動くサンプル（01〜08）
docs/                UI に表示する解説
data/                記憶の保存先（削除すればリセット）
```

## mem0 2.x で押さえておくポイント

- `search()` / `get_all()` のスコープは `filters={"user_id": ...}` で指定する（`user_id=` を直接渡すとエラー）
- `add(infer=True)` は **追加専用**: LLM 呼び出し 1 回で事実を抽出して追加する。既存の記憶を自動で更新・削除はしないので、
  変わった事実への対処はエージェント側で設計する（[docs/05_best_practices.md](docs/05_best_practices.md) を参照）
- `add(infer=False)` は文字列をそのまま保存する（LLM 呼び出しも重複チェックもしない）
- 検索は意味（ベクトル）＋ BM25 ＋ エンティティのハイブリッド。BM25 とエンティティは spaCy の英語モデルを使う（`pip install "mem0ai[nlp]"`）ため、日本語の記憶では主に意味検索が効く

## 補足

- Claude を使う場合、エージェントは `claude-opus-5-5`（effort=low）を使い、安全分類器に拒否されたときは
  サーバーサイド・フォールバック（`fallbacks="default"`）で別モデルが応答を引き継ぎます。不要なら `.env` に `ANTHROPIC_FALLBACKS=0` を書きます。
- 事実の抽出だけを安いモデルにしたい場合は `MEMORY_LLM_MODEL` を設定します（例: `claude-haiku-5-5`、`gpt-5-mini`）。
- 埋め込みモデルごとに `data/` 以下の保存先が分かれます（次元数の違うベクトルは同じ DB に入れられないため）。
