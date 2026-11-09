"""レッスン用の表示ヘルパー"""

from __future__ import annotations

import atexit
import json
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("mem0").setLevel(logging.ERROR)

_SKIP_KEYS = {"hash", "text_lemmatized"}


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n■ {title}\n{'=' * 70}")


def note(text: str) -> None:
    print(f"  💡 {text}")


def show(obj, keys: tuple[str, ...] | None = None) -> None:
    """mem0 の戻り値を見やすく表示する。keys を指定するとその項目だけを表示。"""
    if isinstance(obj, dict) and "results" in obj:
        items = obj["results"]
        print(f"  results: {len(items)} 件")
        for i, item in enumerate(items, 1):
            if keys:
                item = {k: item.get(k) for k in keys if k in item}
            else:
                item = {k: v for k, v in item.items() if k not in _SKIP_KEYS}
            print(f"  [{i}] " + json.dumps(item, ensure_ascii=False, default=str))
        return
    print("  " + json.dumps(obj, ensure_ascii=False, indent=2, default=str).replace("\n", "\n  "))


def fresh_memory(namespace: str, **kwargs):
    """レッスン用に、空の Memory を作る（毎回同じ結果になるよう前回分は消す）。"""
    from .config import build_memory, load_settings

    print(f"[設定] {load_settings().describe()}")
    m = build_memory(namespace, **kwargs)
    m.reset()
    # 終了時にローカル Qdrant を明示的に閉じる（閉じないと終了時に警告が出る）
    atexit.register(m.close)
    return m
