"""
プロバイダ非依存の「ツール呼び出しループ」
=========================================

エージェントの本体は次のループです（どの LLM でも同じ）:

    1. LLM に「システムプロンプト + 会話履歴 + 使えるツール一覧」を送る
    2. LLM がツール呼び出しを返したら → Python 関数を実行し、結果を LLM に返す → 1 へ
    3. LLM が普通のテキストを返したら → それが最終回答

OpenAI / Ollama(OpenAI 互換 API) / Anthropic(Claude) の書き方の違いはこのファイルに閉じ込め、
agent.py からは `ChatLLM.run()` だけを使います。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Settings, load_settings


@dataclass
class Tool:
    """LLM に渡すツール。parameters は JSON Schema。"""

    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]


@dataclass
class RunResult:
    text: str
    tool_events: list[dict] = field(default_factory=list)  # UI 表示用: 呼ばれたツールと結果


def _call_tool(tools: dict[str, Tool], name: str, args: dict) -> tuple[Any, bool]:
    """ツールを実行して (結果, エラーかどうか) を返す。例外は LLM にエラーとして伝える。"""
    tool = tools.get(name)
    if tool is None:
        return {"error": f"unknown tool: {name}"}, True
    try:
        return tool.func(**args), False
    except Exception as e:  # LLM が不正な引数を渡した場合なども含めて、会話を止めずに返す
        return {"error": f"{type(e).__name__}: {e}"}, True


# Claude のサーバーサイド・フォールバック（安全分類器による拒否時に別モデルで再実行）に対応するモデル
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


class ChatLLM:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or load_settings()
        self.model = self.s.agent_model
        if self.s.llm_provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic()
        else:
            from openai import OpenAI

            if self.s.llm_provider == "ollama":
                # Ollama は OpenAI 互換エンドポイントを持っているので同じコードで呼べる
                self.client = OpenAI(base_url=f"{self.s.ollama_base_url}/v1", api_key="ollama")
            else:
                self.client = OpenAI()

    def run(
        self,
        system: str,
        messages: list[dict],
        tools: list[Tool] | None = None,
        max_steps: int = 8,
    ) -> RunResult:
        """messages は [{"role": "user"|"assistant", "content": str}, ...]（最後は user）。"""
        if self.s.llm_provider == "anthropic":
            return self._run_anthropic(system, messages, tools or [], max_steps)
        return self._run_openai(system, messages, tools or [], max_steps)

    # ------------------------------------------------------------------ OpenAI / Ollama
    def _run_openai(self, system, messages, tools, max_steps) -> RunResult:
        tool_map = {t.name: t for t in tools}
        msgs: list[dict] = [{"role": "system", "content": system}, *messages]
        kwargs: dict = {"model": self.model}
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ]
        events: list[dict] = []
        for _ in range(max_steps):
            resp = self.client.chat.completions.create(messages=msgs, **kwargs)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return RunResult(msg.content or "", events)

            msgs.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                }
            )
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    result, is_error = _call_tool(tool_map, tc.function.name, args)
                except json.JSONDecodeError as e:
                    args, result, is_error = {}, {"error": f"invalid JSON arguments: {e}"}, True
                events.append({"tool": tc.function.name, "args": args, "result": result, "error": is_error})
                msgs.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False, default=str)}
                )
        return RunResult("（ツール呼び出しの上限に達しました）", events)

    # ------------------------------------------------------------------ Anthropic (Claude)
    def _run_anthropic(self, system, messages, tools, max_steps) -> RunResult:
        tool_map = {t.name: t for t in tools}
        msgs: list[dict] = list(messages)
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 16000,
            "system": system,
            # チャット用途なので effort は low を既定に（.env の ANTHROPIC_EFFORT で変更可）
            "output_config": {"effort": self.s.anthropic_effort},
        }
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]
        use_fallbacks = self.model in _FALLBACK_MODELS and os.getenv("ANTHROPIC_FALLBACKS", "1") == "1"

        events: list[dict] = []
        for _ in range(max_steps):
            if use_fallbacks:
                resp = self.client.beta.messages.create(
                    messages=msgs,
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    **kwargs,
                )
            else:
                resp = self.client.messages.create(messages=msgs, **kwargs)

            if resp.stop_reason == "refusal":
                return RunResult("（このリクエストにはお応えできませんでした）", events)

            # thinking ブロックなども含め、レスポンスはそのまま履歴に戻すのがルール
            msgs.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text")
                return RunResult(text, events)

            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                result, is_error = _call_tool(tool_map, block.name, dict(block.input or {}))
                events.append({"tool": block.name, "args": block.input, "result": result, "error": is_error})
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                        "is_error": is_error,
                    }
                )
            # 並列ツール呼び出しの結果は 1 つの user メッセージにまとめて返す
            msgs.append({"role": "user", "content": results})
        return RunResult("（ツール呼び出しの上限に達しました）", events)
