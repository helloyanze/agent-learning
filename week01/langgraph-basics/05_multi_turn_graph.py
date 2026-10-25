"""Task 12.5.2: LangGraph + DeepSeek 多轮会话示例（进程内 Checkpointer）。"""

import operator
import os
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, ValidationError


# 1. 初始化 DeepSeek（沿用 llm-basics/.env）
ENV_PATH = Path(__file__).resolve().parent.parent / "llm-basics" / ".env"
load_dotenv(ENV_PATH)

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError(f"未找到 DEEPSEEK_API_KEY，请检查 {ENV_PATH}")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
MODEL = "deepseek-v4-pro"


# 2. 工具及参数校验
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
        "你是计算助手。涉及数学计算时，必须调用提供的工具。"
        "你可以参考本会话之前的计算结果。"
        "有依赖的工具调用须等待前一个工具结果返回。"
        "禁止编造工具执行结果。"
    ),
}


# 3. State：消息历史追加；预算字段在每次 ask() 中重置
class AgentState(TypedDict):
    messages: Annotated[list[dict[str, Any]], operator.add]
    llm_calls: int
    tool_calls: int
    status: Literal["running", "completed", "failed"]
    final_answer: str | None


MAX_LLM_CALLS = 5
MAX_TOOL_CALLS = 5


# 4. LLM 节点
def llm_node(state: AgentState) -> dict[str, Any]:
    if state["llm_calls"] >= MAX_LLM_CALLS:
        return {
            "status": "failed",
            "final_answer": "达到本次最大 LLM 调用次数",
        }

    call_number = state["llm_calls"] + 1
    print(f"\n===== LLM 第 {call_number} 轮 =====")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[SYSTEM_MESSAGE, *state["messages"]],
            tools=TOOLS,
            tool_choice="auto",
        )
    except OpenAIError as exc:
        return {
            "llm_calls": call_number,
            "status": "failed",
            "final_answer": f"LLM API 调用失败：{type(exc).__name__}",
        }

    choice = response.choices[0]
    raw_message = choice.message

    # 截断消息不加入对话历史，避免保留不完整的 tool_calls
    if choice.finish_reason == "length":
        return {
            "llm_calls": call_number,
            "status": "failed",
            "final_answer": "模型输出被截断",
        }

    # 统一保存字典消息，供 Checkpointer 恢复和再次发给 DeepSeek
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": raw_message.content or "",
    }
    reasoning = getattr(raw_message, "reasoning_content", None)
    if reasoning is not None:
        # DeepSeek 带 tools 的思考模式需回传既往 reasoning_content
        assistant_message["reasoning_content"] = reasoning

    if raw_message.tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in raw_message.tool_calls
        ]

    updates: dict[str, Any] = {
        "messages": [assistant_message],
        "llm_calls": call_number,
    }

    if not raw_message.tool_calls:
        if assistant_message["content"]:
            updates["status"] = "completed"
            updates["final_answer"] = assistant_message["content"]
        else:
            updates["status"] = "failed"
            updates["final_answer"] = "模型没有返回有效回答"

    return updates


# 5. Tool 节点
def tool_node(state: AgentState) -> dict[str, Any]:
    assistant_message = state["messages"][-1]
    tool_calls = assistant_message.get("tool_calls", [])

    if state["tool_calls"] + len(tool_calls) > MAX_TOOL_CALLS:
        # 为请求中的每个 tool_call 补齐工具响应，避免下次会话的消息不完整
        return {
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": "工具未执行：已达到本次工具调用预算",
                }
                for call in tool_calls
            ],
            "status": "failed",
            "final_answer": "达到本次最大工具调用次数",
        }

    results: list[dict[str, Any]] = []
    for call in tool_calls:
        name = call["function"]["name"]
        arguments_json = call["function"]["arguments"]
        print(f"调用工具：{name}")
        print(f"工具参数：{arguments_json}")

        try:
            func = TOOL_REGISTRY.get(name)
            if func is None:
                raise ValueError(f"未知工具：{name}")
            args = CalculateArgs.model_validate_json(arguments_json)
            result = str(func(**args.model_dump()))
        except (ValidationError, ValueError) as exc:
            result = f"工具调用失败：{type(exc).__name__}: {exc}"

        print(f"工具结果：{result}")
        results.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            }
        )

    return {
        "messages": results,
        "tool_calls": state["tool_calls"] + len(tool_calls),
    }


# 6. 条件路由（消息统一使用 dict，不使用 message.tool_calls）
def route_after_llm(state: AgentState) -> str:
    if state["status"] != "running":
        return "end"
    last_message = state["messages"][-1]
    return "tools" if last_message.get("tool_calls") else "end"


def route_after_tools(state: AgentState) -> str:
    return "llm" if state["status"] == "running" else "end"


# 7. 构建图，并配置线程内存
builder = StateGraph(AgentState)
builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "llm")
builder.add_conditional_edges(
    "llm", route_after_llm, {"tools": "tools", "end": END}
)
builder.add_conditional_edges(
    "tools", route_after_tools, {"llm": "llm", "end": END}
)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# 8. 连续问答：同一 thread_id 自动保留消息，单次预算清零
def ask(thread_id: str, question: str) -> None:
    print(f"\n用户（{thread_id}）：{question}")
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 20,
    }

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

    print(f"\nAgent：{result['final_answer']}")
    print(f"状态：{result['status']}")
    print(f"本次 LLM 调用：{result['llm_calls']}")
    print(f"本次工具调用：{result['tool_calls']}")


def main() -> None:
    ask("user-a", "请使用 add 工具计算 18+24，告诉我结果。")
    ask("user-a", "把刚才计算得到的结果乘以 2，请使用 multiply 工具计算。")
    ask("user-b", "请使用 add 工具计算 3+4。")

    snapshot_a = graph.get_state({"configurable": {"thread_id": "user-a"}})
    snapshot_b = graph.get_state({"configurable": {"thread_id": "user-b"}})
    print("\n===== Checkpointer 状态 =====")
    print("用户 A 消息条数：", len(snapshot_a.values["messages"]))
    print("用户 A 最近回答：", snapshot_a.values["messages"][-1].get("content"))
    print("用户 B 消息条数：", len(snapshot_b.values["messages"]))


if __name__ == "__main__":
    main()
