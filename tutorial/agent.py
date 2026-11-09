"""
メモリ × AI エージェント の 3 つの設計パターン
=============================================

パターン A: 自動メモリ（Retrieve → Generate → Store）       … MemoryChatAgent
    毎ターン、アプリ側のコードが
      ① ユーザー発話でメモリを検索 → ② システムプロンプトに差し込んで応答 → ③ 会話を mem0 に保存
    を必ず実行します。LLM はメモリの存在を意識しません。
    ◎ 実装が簡単・記憶漏れがない   △ 毎ターン検索/保存のコストがかかる

パターン B: ツールとしてのメモリ（Agentic Memory）           … ToolMemoryAgent
    search_memory / save_memory / update_memory / delete_memory をツールとして渡し、
    「いつ思い出すか・何を覚えるか」を LLM 自身に判断させます。
    ◎ 必要なときだけ動く・訂正や削除もできる   △ LLM の判断次第で覚え漏れ・聞き漏れがある

パターン C: マルチエージェントの共有メモリ                    … MemoryChatAgent(agent_id=...)
    user_id スコープ = 全エージェントで共有する「ユーザーについての事実」
    agent_id スコープ = そのエージェント固有のノウハウ（どのユーザーにも使う）
    を使い分け、専門エージェント間で文脈を引き継ぎます。
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from .llm import ChatLLM, RunResult, Tool


@dataclass
class AgentTurn:
    reply: str
    retrieved: list[dict] = field(default_factory=list)  # 応答前に思い出した記憶
    stored: list[dict] = field(default_factory=list)     # 応答後に保存された記憶
    tool_events: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)          # 各フェーズの所要時間（秒）


def format_memories(memories: list[dict]) -> str:
    if not memories:
        return "（関連する記憶はありません）"
    lines = []
    for m in memories:
        date = (m.get("updated_at") or m.get("created_at") or "")[:10]
        lines.append(f"- {m['memory']}" + (f"（記録日: {date}）" if date else ""))
    return "\n".join(lines)


# ====================================================================== パターン A / C
class MemoryChatAgent:
    BASE_PROMPT = (
        "あなたは親切な日本語アシスタントです。\n"
        "以下の『ユーザーについての記憶』を踏まえ、パーソナライズした回答をしてください。\n"
        "記憶同士が矛盾する場合は記録日が新しいものを優先してください。\n"
        "記憶を使ったことをわざわざ説明する必要はありませんが、不確かなら確認してください。"
    )

    def __init__(
        self,
        memory,
        llm: ChatLLM | None = None,
        *,
        agent_id: str | None = None,
        persona: str | None = None,
        use_memory: bool = True,
        top_k: int = 5,
        threshold: float = 0.1,
        background_store: bool = False,
    ):
        self.memory = memory
        self.llm = llm or ChatLLM()
        self.agent_id = agent_id
        self.persona = persona or self.BASE_PROMPT
        self.use_memory = use_memory
        self.top_k = top_k
        self.threshold = threshold
        # 本番では保存（= LLM 呼び出し）を応答の後ろで非同期に回すとレイテンシが下がる
        self._pool = ThreadPoolExecutor(max_workers=1) if background_store else None
        self.pending: Future | None = None

    # ① Retrieve ---------------------------------------------------------------
    def retrieve(self, user_id: str, query: str) -> tuple[list[dict], list[dict]]:
        user_mems = self.memory.search(
            query, filters={"user_id": user_id}, top_k=self.top_k, threshold=self.threshold
        )["results"]
        agent_mems: list[dict] = []
        if self.agent_id:
            # agent_id だけのスコープ = このエージェント固有のノウハウ（ユーザー横断）
            agent_mems = self.memory.search(
                query, filters={"agent_id": self.agent_id}, top_k=3, threshold=self.threshold
            )["results"]
        return user_mems, agent_mems

    # ② Generate ----------------------------------------------------------------
    def build_system_prompt(self, user_mems: list[dict], agent_mems: list[dict]) -> str:
        prompt = self.persona
        if self.use_memory:
            prompt += f"\n\n## ユーザーについての記憶\n{format_memories(user_mems)}"
            if self.agent_id:
                prompt += f"\n\n## あなた自身のノウハウ\n{format_memories(agent_mems)}"
        return prompt

    # ③ Store -------------------------------------------------------------------
    def store(self, user_id: str, user_msg: str, reply: str) -> list[dict]:
        conversation = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ]
        metadata = {"source_agent": self.agent_id} if self.agent_id else None
        # infer=True（既定）: LLM が会話から「長期的に役立つ事実」だけを抜き出して保存する
        return self.memory.add(conversation, user_id=user_id, metadata=metadata)["results"]

    def chat(self, user_id: str, message: str, history: list[dict] | None = None) -> AgentTurn:
        history = history or []
        t0 = time.perf_counter()
        user_mems, agent_mems = self.retrieve(user_id, message) if self.use_memory else ([], [])
        t1 = time.perf_counter()

        system = self.build_system_prompt(user_mems, agent_mems)
        result: RunResult = self.llm.run(system, [*history, {"role": "user", "content": message}])
        t2 = time.perf_counter()

        stored: list[dict] = []
        if self.use_memory:
            if self._pool:
                self.pending = self._pool.submit(self.store, user_id, message, result.text)
            else:
                stored = self.store(user_id, message, result.text)
        t3 = time.perf_counter()

        return AgentTurn(
            reply=result.text,
            retrieved=user_mems + agent_mems,
            stored=stored,
            tool_events=result.tool_events,
            timings={"retrieve": t1 - t0, "generate": t2 - t1, "store": t3 - t2},
        )


# ====================================================================== パターン B
class ToolMemoryAgent:
    SYSTEM = (
        "あなたは長期記憶を持つ日本語アシスタントです。記憶は次のツールで操作します。\n"
        "- search_memory: ユーザーの好み・状況・過去の話題が回答に関係しそうなら、回答前に必ず検索する\n"
        "- save_memory: 今後も役立つユーザーの事実（好み、属性、目標、予定など）が出たら、"
        "簡潔な1文（主語は「ユーザーは」）で保存する。雑談や一時的な情報は保存しない\n"
        "- update_memory: 保存済みの事実が変わった（引っ越した、好みが変わった等）なら、"
        "search_memory で id を調べて更新する\n"
        "- delete_memory: ユーザーが忘れてほしいと言った事実は削除する\n"
        "ツールの使用はユーザーに逐一説明しなくて構いません。"
    )

    def __init__(self, memory, llm: ChatLLM | None = None):
        self.memory = memory
        self.llm = llm or ChatLLM()

    def _tools(self, user_id: str) -> list[Tool]:
        m = self.memory

        def search_memory(query: str, limit: int = 5):
            res = m.search(query, filters={"user_id": user_id}, top_k=limit)["results"]
            return [{"id": r["id"], "memory": r["memory"], "score": round(r.get("score") or 0, 3)} for r in res]

        def save_memory(fact: str):
            # LLM がすでに「何を覚えるか」を決めているので infer=False でそのまま保存する
            res = m.add(fact, user_id=user_id, infer=False, metadata={"saved_by": "tool"})["results"]
            return {"saved": [r["memory"] for r in res]}

        def update_memory(memory_id: str, new_fact: str):
            return m.update(memory_id, new_fact)

        def delete_memory(memory_id: str):
            return m.delete(memory_id)

        str_prop = {"type": "string"}
        return [
            Tool(
                "search_memory",
                "ユーザーについての長期記憶を意味検索する。",
                {
                    "type": "object",
                    "properties": {"query": {**str_prop, "description": "検索したい内容"},
                                   "limit": {"type": "integer", "description": "最大件数", "default": 5}},
                    "required": ["query"],
                },
                search_memory,
            ),
            Tool(
                "save_memory",
                "ユーザーについての新しい事実を長期記憶に保存する。",
                {"type": "object", "properties": {"fact": {**str_prop, "description": "保存する事実（1文）"}},
                 "required": ["fact"]},
                save_memory,
            ),
            Tool(
                "update_memory",
                "既存の記憶を新しい内容で置き換える。",
                {"type": "object",
                 "properties": {"memory_id": str_prop, "new_fact": str_prop},
                 "required": ["memory_id", "new_fact"]},
                update_memory,
            ),
            Tool(
                "delete_memory",
                "記憶を削除する。",
                {"type": "object", "properties": {"memory_id": str_prop}, "required": ["memory_id"]},
                delete_memory,
            ),
        ]

    def chat(self, user_id: str, message: str, history: list[dict] | None = None) -> AgentTurn:
        t0 = time.perf_counter()
        result = self.llm.run(self.SYSTEM, [*(history or []), {"role": "user", "content": message}],
                              tools=self._tools(user_id))
        retrieved = [r for e in result.tool_events if e["tool"] == "search_memory" and not e["error"]
                     for r in e["result"]]
        stored = [{"memory": f, "event": "ADD"} for e in result.tool_events
                  if e["tool"] == "save_memory" and not e["error"] for f in e["result"]["saved"]]
        stored += [{"memory": e["args"].get("new_fact"), "event": "UPDATE"} for e in result.tool_events
                   if e["tool"] == "update_memory" and not e["error"]]
        stored += [{"memory": e["args"].get("memory_id"), "event": "DELETE"} for e in result.tool_events
                   if e["tool"] == "delete_memory" and not e["error"]]
        return AgentTurn(result.text, retrieved, stored, result.tool_events,
                         {"total": time.perf_counter() - t0})
