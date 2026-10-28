"""Task 14.2: DeepSeek + LangGraph + SQLite Agent service.

This module owns the Agent graph; FastAPI routing lives in main.py.
The SQLite checkpointer's lifetime is managed by FastAPI lifespan.
"""

import operator
import os
import threading
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, ValidationError

# 兼容用户此前的学习目录，不移动或复制 .env 文件。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / "week01" / "llm-basics" / ".env")

MODEL = "deepseek-v4-pro"
MAX_LLM_CALLS = 5
MAX_TOOL_CALLS = 5


class CalculateArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    a: int
    b: int


def add(a: int, b: int) -> int:
    return a + b


def multiply(a: int, b: int) -> int:
    return a * b


TOOL_REGISTRY = {"add": add, "multiply": multiply}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": CalculateArgs.model_json_schema(),
        },
    }
    for name, description in [
        ("add", "计算两个整数的和"),
        ("multiply", "计算两个整数的乘积"),
    ]
]

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "你是计算助手。计算加法或乘法时，请使用提供的工具。"
        "后续提问可以引用当前会话中之前的工具结果。"
        "必须等待工具返回真实结果，不能编造计算结果。"
    ),
}


class AgentState(TypedDict):
    messages: Annotated[list[dict[str, Any]], operator.add]
    llm_calls: int
    tool_calls: int
    status: Literal["running", "completed", "failed"]
    final_answer: str | None


def build_graph(client: OpenAI, checkpointer: SqliteSaver):
    """构建图，避免在模块导入时创建 API 请求或 SQLite 连接。"""

    def llm_node(state: AgentState) -> dict[str, Any]:
        if state["llm_calls"] >= MAX_LLM_CALLS:
            return {
                "status": "failed",
                "final_answer": "达到最大 LLM 调用次数",
            }

        attempt = state["llm_calls"] + 1
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[SYSTEM_MESSAGE, *state["messages"]],
                tools=TOOLS,
                tool_choice="auto",
            )
        except OpenAIError as exc:
            return {
                "llm_calls": attempt,
                "status": "failed",
                "final_answer": f"LLM API 调用失败：{type(exc).__name__}",
            }

        choice = response.choices[0]
        raw = choice.message

        if choice.finish_reason == "length":
            return {
                "llm_calls": attempt,
                "status": "failed",
                "final_answer": "模型输出被截断",
            }

        msg: dict[str, Any] = {
            "role": "assistant",
            "content": raw.content or "",
        }
        reasoning = getattr(raw, "reasoning_content", None)
        if reasoning is not None:
            msg["reasoning_content"] = reasoning

        if raw.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in raw.tool_calls
            ]

        updates: dict[str, Any] = {
            "messages": [msg],
            "llm_calls": attempt,
        }
        if not raw.tool_calls:
            if msg["content"]:
                updates.update(status="completed", final_answer=msg["content"])
            else:
                updates.update(status="failed", final_answer="模型未返回有效内容")
        return updates

    def tool_node(state: AgentState) -> dict[str, Any]:
        calls = state["messages"][-1].get("tool_calls") or []
        exceeded = state["tool_calls"] + len(calls) > MAX_TOOL_CALLS
        results = []

        for call in calls:
            name = call["function"]["name"]
            if exceeded:
                outcome = "工具未执行：达到最大工具调用次数"
            else:
                try:
                    func = TOOL_REGISTRY.get(name)
                    if func is None:
                        raise ValueError(f"未知工具：{name}")
                    args = CalculateArgs.model_validate_json(
                        call["function"]["arguments"]
                    )
                    outcome = str(func(**args.model_dump()))
                except (ValueError, ValidationError) as exc:
                    outcome = f"工具调用失败：{exc}"
                except Exception as exc:
                    # 当前只有无外部副作用的数学工具；生产环境应专门处理意外异常。
                    outcome = f"工具内部异常：{type(exc).__name__}"

            results.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": outcome,
            })

        # 即使拒绝执行，也配对每一个 tool_call_id，以保留合法消息历史。
        updates: dict[str, Any] = {"messages": results}
        if exceeded:
            updates.update(status="failed", final_answer="达到最大工具调用次数")
        else:
            updates["tool_calls"] = state["tool_calls"] + len(calls)
        return updates

    def after_llm(state: AgentState) -> str:
        if state["status"] != "running":
            return "end"
        return "tools" if state["messages"][-1].get("tool_calls") else "end"

    def after_tools(state: AgentState) -> str:
        return "llm" if state["status"] == "running" else "end"

    builder = StateGraph(AgentState)
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", after_llm, {"tools": "tools", "end": END})
    builder.add_conditional_edges("tools", after_tools, {"llm": "llm", "end": END})
    return builder.compile(checkpointer=checkpointer)


class AgentService:
    def __init__(self, graph):
        self.graph = graph
        # 教学用同步 SQLite Saver 适合低并发。串行处理请求，避免同一
        # thread_id 同时写入造成历史分叉；不用于高并发生产环境。
        self._lock = threading.Lock()

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 20,
        }

    def ask(self, thread_id: str, message: str) -> dict[str, Any]:
        with self._lock:
            result = self.graph.invoke(
                {
                    "messages": [{"role": "user", "content": message}],
                    "llm_calls": 0,
                    "tool_calls": 0,
                    "status": "running",
                    "final_answer": None,
                },
                config=self._config(thread_id),
            )

        return {
            "thread_id": thread_id,
            "answer": result["final_answer"] or "",
            "status": result["status"],
            "llm_calls": result["llm_calls"],
            "tool_calls": result["tool_calls"],
            "history_count": len(result["messages"]),
        }

    def history(self, thread_id: str) -> list[dict[str, str]]:
        with self._lock:
            snapshot = self.graph.get_state(self._config(thread_id))
        messages = snapshot.values.get("messages", [])
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg.get("role") in {"user", "assistant"}
            and msg.get("content")
            and not msg.get("tool_calls")
        ]


def create_service(checkpointer: SqliteSaver) -> AgentService:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError(
            "找不到 DEEPSEEK_API_KEY，请检查 week01/llm-basics/.env"
        )
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    return AgentService(build_graph(client, checkpointer))
