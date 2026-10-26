"""Task 13.2: DeepSeek + LangGraph + SQLite persistence across processes.

Examples (run each as a separate Python process):
    python 07_sqlite_deepseek_agent.py --thread user-a --question "请使用 add 计算18+24"
    python 07_sqlite_deepseek_agent.py --thread user-a --question "把刚才的结果乘以2，请使用 multiply"
    python 07_sqlite_deepseek_agent.py --thread user-b --question "请使用 add 计算3+4"
"""

import argparse
import operator
import os
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, ValidationError

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / "llm-basics" / ".env")

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("找不到 DEEPSEEK_API_KEY，请检查 llm-basics/.env")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
MODEL = "deepseek-v4-pro"
MAX_LLM_CALLS = 5  # 每次用户提问单独计算
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
        "你是计算助手。计算加法或乘法时，必须调用提供的工具。"
        "需要使用之前的结果时，先查阅同一会话中此前的工具输出。"
        "不要编造计算结果，也不要使用其他会话的信息。"
    ),
}


class AgentState(TypedDict):
    messages: Annotated[list[dict[str, Any]], operator.add]
    llm_calls: int
    tool_calls: int
    status: Literal["running", "completed", "failed"]
    final_answer: str | None


def llm_node(state: AgentState) -> dict:
    if state["llm_calls"] >= MAX_LLM_CALLS:
        return {"status": "failed", "final_answer": "达到最大 LLM 调用次数"}

    attempt = state["llm_calls"] + 1
    print(f"  LLM 第 {attempt} 轮")
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

    # 截断的消息不加入历史，避免保存不完整的工具调用。
    if choice.finish_reason == "length":
        return {
            "llm_calls": attempt,
            "status": "failed",
            "final_answer": "模型输出被截断",
        }

    message: dict[str, Any] = {
        "role": "assistant",
        "content": raw.content or "",
    }
    reasoning = getattr(raw, "reasoning_content", None)
    if reasoning is not None:
        # DeepSeek 携带 tools 时要求回传历史 reasoning_content。
        message["reasoning_content"] = reasoning

    if raw.tool_calls:
        message["tool_calls"] = [
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
        "messages": [message],
        "llm_calls": attempt,
    }
    if not raw.tool_calls:
        if message["content"]:
            updates.update(status="completed", final_answer=message["content"])
        else:
            updates.update(status="failed", final_answer="模型没有返回有效内容")
    return updates


def tool_node(state: AgentState) -> dict:
    calls = state["messages"][-1].get("tool_calls", [])
    over_budget = state["tool_calls"] + len(calls) > MAX_TOOL_CALLS
    results: list[dict[str, str]] = []

    for call in calls:
        name = call["function"]["name"]
        if over_budget:
            result = "工具未执行：达到最大工具调用次数"
        else:
            try:
                args = CalculateArgs.model_validate_json(call["function"]["arguments"])
                function = TOOL_REGISTRY.get(name)
                if function is None:
                    raise ValueError(f"未知工具：{name}")
                result = str(function(**args.model_dump()))
            except (ValueError, ValidationError) as exc:
                result = f"工具调用失败：{exc}"
            except Exception as exc:
                # 教学版工具无外部副作用；意外异常仍作为错误结果返回。
                result = f"工具内部异常：{type(exc).__name__}"

        print(f"  工具 {name} → {result}")
        results.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    # 即使超出预算，也为每个 tool_call_id 配一个 tool 消息，
    # 避免后续用户提问时出现未配对的工具调用历史。
    updates: dict[str, Any] = {"messages": results}
    if over_budget:
        updates.update(status="failed", final_answer="达到最大工具调用次数")
    else:
        updates["tool_calls"] = state["tool_calls"] + len(calls)
    return updates


def route_after_llm(state: AgentState) -> str:
    if state["status"] != "running":
        return "end"
    return "tools" if state["messages"][-1].get("tool_calls") else "end"


def route_after_tools(state: AgentState) -> str:
    return "llm" if state["status"] == "running" else "end"


builder = StateGraph(AgentState)
builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", route_after_llm, {"tools": "tools", "end": END})
builder.add_conditional_edges("tools", route_after_tools, {"llm": "llm", "end": END})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="支持 SQLite 记忆的 DeepSeek 计算 Agent"
    )
    parser.add_argument("--thread", default="user-a", help="会话 ID，例如 user-a")
    parser.add_argument("--question", required=True, help="本次用户提问")
    args = parser.parse_args()

    db_path = HERE / "task13_agent_checkpoints.sqlite"
    config = {
        "configurable": {"thread_id": args.thread},
        "recursion_limit": 20,
    }

    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        saved = graph.get_state(config).values
        print(f"会话：{args.thread}")
        print(f"本次调用前的历史消息数：{len(saved.get('messages', []))}")
        print(f"问题：{args.question}\n")

        # messages 追加到历史；其他普通字段在每次提问时重置。
        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": args.question}],
                "llm_calls": 0,
                "tool_calls": 0,
                "status": "running",
                "final_answer": None,
            },
            config=config,
        )

        print("\n===== 本次结果 =====")
        print("状态：", result["status"])
        print("回答：", result["final_answer"])
        print("本次 LLM 调用：", result["llm_calls"])
        print("本次工具调用：", result["tool_calls"])
        print("保存后的历史消息数：", len(result["messages"]))
        print("SQLite 文件：", db_path)


if __name__ == "__main__":
    main()
