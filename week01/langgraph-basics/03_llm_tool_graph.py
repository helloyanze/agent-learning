import os
import operator
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, ValidationError
from langgraph.graph import StateGraph, START, END

# =========================
# 1. 初始化 DeepSeek
# =========================

env_path = Path(__file__).resolve().parent.parent / "llm-basics" / ".env"

load_dotenv(env_path)

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("没有找到 DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


# =========================
# 2. 定义工具
# =========================


class CalculateArgs(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    a: int
    b: int


def add(a: int, b: int) -> int:
    return a + b


def multiply(a: int, b: int) -> int:
    return a * b


TOOL_REGISTRY = {
    "add": add,
    "multiply": multiply,
}

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


# =========================
# 3. 定义 LangGraph State
# =========================


class AgentState(TypedDict):
    messages: Annotated[list[Any], operator.add]
    llm_calls: int
    tool_calls: int
    status: Literal["running", "completed", "failed"]
    final_answer: str | None


MAX_LLM_CALLS = 5
MAX_TOOL_CALLS = 5


# =========================
# 4. LLM Node
# =========================


def llm_node(state: AgentState) -> dict:

    if state["llm_calls"] >= MAX_LLM_CALLS:
        return {
            "status": "failed",
            "final_answer": "达到最大 LLM 调用次数",
        }

    round_num = state["llm_calls"] + 1
    print(f"\n===== LLM 第 {round_num} 轮 =====")

    try:
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=state["messages"],
            tools=TOOLS,
            tool_choice="auto",
        )

    except OpenAIError as e:
        return {
            "llm_calls": round_num,
            "status": "failed",
            "final_answer": (f"LLM API 调用失败：{type(e).__name__}"),
        }

    choice = response.choices[0]
    message = choice.message

    updates = {
        "messages": [message],
        "llm_calls": round_num,
    }

    if choice.finish_reason == "length":
        updates["status"] = "failed"
        updates["final_answer"] = "模型输出被截断"

    elif not message.tool_calls:
        if message.content:
            updates["status"] = "completed"
            updates["final_answer"] = message.content
        else:
            updates["status"] = "failed"
            updates["final_answer"] = "模型没有返回有效内容"

    return updates


# =========================
# 5. Tool Node
# =========================


def tool_node(state: AgentState) -> dict:

    message = state["messages"][-1]
    tool_calls = message.tool_calls or []

    if state["tool_calls"] + len(tool_calls) > MAX_TOOL_CALLS:
        return {
            "status": "failed",
            "final_answer": "达到最大工具调用次数",
        }

    results = []

    for tool_call in tool_calls:

        name = tool_call.function.name
        arguments_json = tool_call.function.arguments

        print(f"调用工具：{name}")
        print(f"工具参数：{arguments_json}")

        try:
            function = TOOL_REGISTRY.get(name)

            if function is None:
                raise ValueError(f"未知工具：{name}")

            args = CalculateArgs.model_validate_json(arguments_json)

            result = str(function(**args.model_dump()))

        except (ValueError, ValidationError) as e:
            result = f"工具调用失败：{e}"

        print(f"工具结果：{result}")

        results.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )

    return {
        "messages": results,
        "tool_calls": state["tool_calls"] + len(tool_calls),
    }


# =========================
# 6. Conditional Edge
# =========================


def route_after_llm(state: AgentState) -> str:

    if state["status"] != "running":
        return "end"

    message = state["messages"][-1]

    if message.tool_calls:
        return "tools"

    return "end"


def route_after_tools(state: AgentState) -> str:

    if state["status"] == "running":
        return "llm"

    return "end"


# =========================
# 7. 构建 Graph
# =========================

builder = StateGraph(AgentState)

builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "llm")

builder.add_conditional_edges(
    "llm",
    route_after_llm,
    {
        "tools": "tools",
        "end": END,
    },
)

builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "llm": "llm",
        "end": END,
    },
)

graph = builder.compile()


# =========================
# 8. 执行
# =========================


def main() -> None:

    initial_state: AgentState = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是计算助手，必须使用工具完成计算。"
                    "存在前后依赖时，要等待前一个工具结果。"
                    "不要编造工具执行结果。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请先使用 add 计算 18+24，"
                    "得到结果后再使用 multiply 乘以2，"
                    "最后给出答案。"
                ),
            },
        ],
        "llm_calls": 0,
        "tool_calls": 0,
        "status": "running",
        "final_answer": None,
    }

    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": 20},
    )

    print("\n===== 执行结果 =====")
    print(f"状态：{final_state['status']}")
    print(f"LLM 调用：{final_state['llm_calls']}")
    print(f"工具调用：{final_state['tool_calls']}")
    print(f"最终回答：{final_state['final_answer']}")


if __name__ == "__main__":
    main()
