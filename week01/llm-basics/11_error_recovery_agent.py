import os
from pathlib import Path
from dataclasses import asdict

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from agent_state import (
    AgentState,
    start_round,
    start_tool,
    complete,
    fail,
)
from tool_definitions import TOOLS
from tool_executor import execute_tool

# 1. 初始化 DeepSeek
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("没有找到 DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


# 2. Agent Loop
def run_agent(
    user_input: str,
    state: AgentState,
) -> str | None:

    messages = [
        {
            "role": "system",
            "content": (
                "你是计算助手。"
                "遇到计算任务时调用提供的工具。"
                "存在依赖关系时，必须等待前一个工具结果。"
                "根据真实工具结果回答，不要编造。"
            ),
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    while state.status == "running":

        # 检查是否允许开始新一轮
        if not start_round(state):
            break

        print(f"\n===== 第 {state.rounds} 轮 =====")

        try:
            response = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

        except OpenAIError as e:
            fail(state, f"LLM API 调用失败：{type(e).__name__}")
            print(f"API 异常：{e}")
            break

        choice = response.choices[0]
        message = choice.message

        if choice.finish_reason == "length":
            fail(state, "模型输出被截断")
            break

        messages.append(message)

        # 模型不再调用工具：准备结束
        if not message.tool_calls:

            if not message.content:
                fail(state, "模型没有返回有效回答")
                break

            complete(state)

            print("\nAgent 最终回答：")
            print(message.content)

            return message.content

        # 处理当前轮所有工具调用
        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            # 每次调用工具前检查预算
            if not start_tool(state, tool_name):
                print("工具预算不足，停止执行")
                return None

            print(f"调用工具：{tool_name}")

            try:
                result = execute_tool(
                    tool_name,
                    tool_call.function.arguments,
                )

                print(f"工具执行成功：{result}")

            except (ValueError, ValidationError) as e:
                # 参数或工具名称错误：允许模型纠正
                error_message = f"{type(e).__name__}: {e}"

                state.last_error = error_message
                state.events.append(f"工具 {tool_name} 参数校验失败")

                result = (
                    f"工具调用失败：{error_message}。"
                    "请检查参数类型及工具名称，"
                    "如有必要，重新调用正确的工具。"
                )

            except Exception as e:
                # 教学阶段：未预期的工具内部错误视为致命错误
                fail(state, f"工具内部错误：{type(e).__name__}")
                print(f"工具内部异常：{e}")
                return None

            print(f"工具结果：{result}")

            # 保存工具结果
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    return None


# 3. 主程序
def main() -> None:

    state = AgentState(
        max_rounds=3,
        max_tool_calls=2,
    )

    user_input = (
        "请先调用 add 计算 18 + 24，"
        "等结果返回后，再调用 multiply 乘以 2。"
        "最后根据实际工具结果回答。"
    )

    result = run_agent(user_input, state)

    print("\n===== Agent State =====")
    print(asdict(state))

    print("\n===== 执行日志 =====")
    for event in state.events:
        print(event)

    print("\n===== 最终结果 =====")
    print(result)


if __name__ == "__main__":
    main()
