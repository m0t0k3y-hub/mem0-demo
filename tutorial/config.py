"""
mem0 チュートリアル共通設定
==========================

mem0 の `Memory` は 3 つの部品を組み合わせて動きます。

    ┌──────────────┐   事実の抽出    ┌──────────────┐
    │     LLM      │ ─────────────▶ │   Embedder   │  テキスト → ベクトル
    └──────────────┘                └──────┬───────┘
                                           ▼
                                   ┌──────────────┐
                                   │ Vector Store │  (Qdrant など) 保存・検索
                                   └──────────────┘
                                   + history.db (SQLite: 変更履歴)

このモジュールは .env の設定から、この 3 つを決めて `Memory` を作ります。

  LLM_PROVIDER      = openai | anthropic | ollama   (未指定なら API キーから自動判定)
  EMBEDDER_PROVIDER = openai | fastembed | ollama   (未指定なら自動判定)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# mem0 は匿名テレメトリを送信します。チュートリアルではオフにしておきます。
os.environ.setdefault("MEM0_TELEMETRY", "False")

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data")).resolve()

# 埋め込みモデルごとのベクトル次元数（ベクトルストアの作成時に必要）
_EMBED_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "intfloat/multilingual-e5-large": 1024,
    "nomic-embed-text": 768,
    "bge-m3": 1024,
}


@dataclass(frozen=True)
class Settings:
    llm_provider: str          # エージェントと mem0 の両方が使う LLM のプロバイダ
    agent_model: str           # エージェント（会話）に使うモデル
    memory_model: str          # mem0 が「事実の抽出」に使うモデル
    embedder_provider: str
    embedder_model: str
    embedding_dims: int
    ollama_base_url: str
    anthropic_effort: str

    def describe(self) -> str:
        return (
            f"LLM: {self.llm_provider} (agent={self.agent_model}, memory={self.memory_model}) / "
            f"Embedder: {self.embedder_provider} ({self.embedder_model}, {self.embedding_dims}次元)"
        )


def _auto_llm_provider() -> str:
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "ollama"


def load_settings() -> Settings:
    llm = (os.getenv("LLM_PROVIDER") or _auto_llm_provider()).lower()
    if llm not in ("openai", "anthropic", "ollama"):
        raise ValueError(f"LLM_PROVIDER は openai / anthropic / ollama のいずれかです: {llm}")

    if llm == "openai":
        agent_model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    elif llm == "anthropic":
        agent_model = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
    else:
        agent_model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    memory_model = os.getenv("MEMORY_LLM_MODEL") or agent_model

    # Anthropic には埋め込み API が無いため、ローカルで動く fastembed を既定にする
    default_embedder = "openai" if llm == "openai" else "fastembed"
    embedder = (os.getenv("EMBEDDER_PROVIDER") or default_embedder).lower()
    if embedder == "openai":
        embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    elif embedder == "fastembed":
        # 日本語を扱えるよう多言語モデルを既定にしている（初回に約 200MB ダウンロード）
        embed_model = os.getenv(
            "FASTEMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    elif embedder == "ollama":
        embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    else:
        raise ValueError(f"EMBEDDER_PROVIDER は openai / fastembed / ollama のいずれかです: {embedder}")

    dims = int(os.getenv("EMBEDDING_DIMS") or _EMBED_DIMS.get(embed_model, 0))
    if not dims:
        raise ValueError(f"{embed_model} の次元数が不明です。EMBEDDING_DIMS を .env に設定してください。")

    return Settings(
        llm_provider=llm,
        agent_model=agent_model,
        memory_model=memory_model,
        embedder_provider=embedder,
        embedder_model=embed_model,
        embedding_dims=dims,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        anthropic_effort=os.getenv("ANTHROPIC_EFFORT", "low"),
    )


def _is_openai_reasoning_model(model: str) -> bool | None:
    """temperature 等を受け付けない OpenAI の推論モデルかどうか。

    .env の OPENAI_REASONING_MODEL=1/0 で上書きできる。判断できなければ None（mem0 の判定に任せる）。
    """
    override = os.getenv("OPENAI_REASONING_MODEL")
    if override is not None and override.strip() != "":
        return override.strip().lower() in ("1", "true", "yes")
    base = model.lower().rsplit("/", 1)[-1]
    if base in ("gpt-5", "gpt-5-mini", "gpt-5-nano") or base.startswith(("gpt-5-mini-", "gpt-5-nano-", "gpt-5-20")):
        return True
    if base.startswith(("o1", "o3", "o4")):
        return True
    return None


def build_mem0_config(
    namespace: str,
    settings: Settings | None = None,
    custom_instructions: str | None = None,
) -> dict:
    """`Memory.from_config()` に渡す設定 dict を作る。

    namespace ごとに別のデータディレクトリを使います。
    （ローカル Qdrant は 1 つのディレクトリを同時に 1 プロセスしか開けないため、
      レッスンごと・アプリごとに分けています）
    """
    s = settings or load_settings()
    # 埋め込みモデルを変えるとベクトルの次元が変わり、既存のデータと混ぜられない。
    # そのためモデルごとに保存先を分ける。
    embed_slug = f"{s.embedder_provider}-{s.embedder_model}".replace("/", "_").replace(":", "_")
    base = DATA_DIR / namespace / embed_slug
    base.mkdir(parents=True, exist_ok=True)

    # --- LLM: 会話から「覚えるべき事実」を抽出するのに使う ---
    if s.llm_provider == "openai":
        llm = {"provider": "openai", "config": {"model": s.memory_model}}
        # mem0 の判定は "gpt-5" 完全一致などに限られ、既定の gpt-5-mini を推論モデルと認識できない。
        # その結果 temperature=0.1 を送って 400 エラーになるため、明示的に指定する。
        reasoning = _is_openai_reasoning_model(s.memory_model)
        if reasoning is not None:
            llm["config"]["is_reasoning_model"] = reasoning
    elif s.llm_provider == "anthropic":
        llm = {
            "provider": "anthropic",
            "config": {
                "model": s.memory_model,
                "max_tokens": 8000,
                # 新しい Claude モデルは temperature 等を受け付けないので送らない
                "enable_sampling_parameters": False,
            },
        }
    else:
        llm = {
            "provider": "ollama",
            "config": {"model": s.memory_model, "ollama_base_url": s.ollama_base_url, "max_tokens": 4000},
        }

    # --- Embedder: テキストをベクトルに変換（意味検索に使う） ---
    embed_cfg: dict = {"model": s.embedder_model, "embedding_dims": s.embedding_dims}
    if s.embedder_provider == "ollama":
        embed_cfg["ollama_base_url"] = s.ollama_base_url
    if s.embedder_provider == "openai":
        embed_cfg.pop("embedding_dims")  # OpenAI は既定次元のまま使う
    embedder = {"provider": s.embedder_provider, "config": embed_cfg}

    # --- Vector Store: ローカルファイルの Qdrant（サーバー不要） ---
    vector_store = {
        "provider": "qdrant",
        "config": {
            "collection_name": "memories",
            "path": str(base / "qdrant"),
            "embedding_model_dims": s.embedding_dims,
            "on_disk": True,  # プロセスを終了しても記憶が残る
        },
    }

    config = {
        "llm": llm,
        "embedder": embedder,
        "vector_store": vector_store,
        "history_db_path": str(base / "history.db"),
    }
    if custom_instructions:
        config["custom_instructions"] = custom_instructions
    return config


def build_memory(namespace: str, custom_instructions: str | None = None):
    """設定済みの mem0 `Memory` インスタンスを返す。"""
    from mem0 import Memory

    return Memory.from_config(build_mem0_config(namespace, custom_instructions=custom_instructions))
