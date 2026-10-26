"""Task 13.3: interactive DeepSeek/LangGraph CLI with persistent SQLite threads.

Run: python 08_interactive_agent.py --thread study-user
Requires: llm-basics/.env with DEEPSEEK_API_KEY and the installed dependencies.
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


def thread_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 20,
    }


def history_messages(graph, thread_id: str) -> list[dict[str, Any]]:
    snapshot = graph.get_state(thread_config(thread_id))
    return snapshot.values.get("messages", [])


def print_history(graph, thread_id: str) -> None:
    messages = history_messages(graph, thread_id)
    print(f"\n===== 会话 {thread_id}：{len(messages)} 条保存的消息 =====")
    if not messages:
        print("暂无历史")
        return

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            print(f"你：{content}")
        elif role == "assistant" and content and not message.get("tool_calls"):
            print(f"Agent：{content}")
    print("（工具调用和工具结果已保存，但在简要历史中省略。）")


def print_help() -> None:
    print("可用命令：")
    print("  /help             查看命令")
    print("  /history          查看当前会话的问答历史")
    print("  /status           查看当前会话的保存状态")
    print("  /thread 新会话ID   切换会话；不同 ID 的历史相互隔离")
    print("  /exit             退出程序（SQLite 历史仍保留）")


def ask(graph, thread_id: str, question: str) -> None:
    config = thread_config(thread_id)

    # 仅追加本次用户消息；数值预算字段每轮重置。
    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": question}],
            "llm_calls": 0,
            "tool_calls": 0,
            "status": "running",
            "final_answer": None,
        },
        config=config,
    )

    print("\n===== Agent 回答 =====")
    print(result["final_answer"] or "（没有得到回答）")
    print(
        f"状态：{result['status']} | "
        f"本次 LLM 请求：{result['llm_calls']} | "
        f"本次工具调用：{result['tool_calls']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="支持 SQLite 记忆的交互式 DeepSeek Agent"
    )
    parser.add_argument("--thread", default="study-user", help="会话 ID")
    args = parser.parse_args()
    thread_id = args.thread.strip()
    if not thread_id:
        parser.error("--thread 不能为空")

    db_path = HERE / "task13_agent_checkpoints.sqlite"

    # 同一 Python 进程中只打开一次 SQLite 连接。
    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        print("DeepSeek 交互式计算 Agent 已启动")
        print(f"当前会话：{thread_id}")
        print(f"已保存的消息：{len(history_messages(graph, thread_id))} 条")
        print(f"SQLite 数据文件：{db_path}")
        print_help()

        while True:
            try:
                user_input = input(f"\n[{thread_id}] 你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出；历史仍保存在 SQLite 中。")
                break

            if not user_input:
                continue
            if user_input == "/exit":
                print("已退出；历史仍保存在 SQLite 中。")
                break
            if user_input == "/help":
                print_help()
                continue
            if user_input == "/history":
                print_history(graph, thread_id)
                continue
            if user_input == "/status":
                snapshot = graph.get_state(thread_config(thread_id)).values
                print(f"当前会话：{thread_id}")
                print(f"历史消息数：{len(snapshot.get('messages', []))}")
                print(f"最近一次执行状态：{snapshot.get('status', '暂无记录')}")
                continue
            if user_input.startswith("/thread"):
                parts = user_input.split(maxsplit=1)
                if len(parts) != 2 or not parts[1].strip():
                    print("用法：/thread 新会话ID")
                    continue
                thread_id = parts[1].strip()
                print(
                    f"已切换会话：{thread_id}，历史消息 {len(history_messages(graph, thread_id))} 条"
                )
                continue
            if user_input.startswith("/"):
                print("未知命令，输入 /help 查看帮助。")
                continue

            ask(graph, thread_id, user_input)


if __name__ == "__main__":
    main()
