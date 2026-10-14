import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from tool_definitions import TOOLS
from tool_executor import execute_tool

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("没有找到 DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


def main() -> None:

    messages = [
        {
            "role": "user",
            "content": "请使用工具计算 18 加 24 等于多少？",
        }
    ]

    # 1. 调用 LLM
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        tools=TOOLS,
    )

    message = response.choices[0].message

    if not message.tool_calls:
        print("模型没有调用工具")
        print(message.content)
        return

    # 2. 保存模型的 Tool Call 消息
    messages.append(message)

    # 3. 处理工具调用
    for tool_call in message.tool_calls:

        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments

        print(f"工具名称：{tool_name}")
        print(f"工具参数：{arguments}")

        try:
            result = execute_tool(
                tool_name,
                arguments,
            )

        except (ValidationError, ValueError) as e:
            result = f"工具调用失败：{e}"

        print(f"工具执行结果：{result}")

        # 4. 将结果返回给 LLM
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )

    # 5. 再次调用 LLM
    final_response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
    )

    print("最终回答：")
    print(final_response.choices[0].message.content)


if __name__ == "__main__":
    main()


"""
概念	职责
Tool Schema	告诉 LLM 有哪些工具、需要什么参数
Tool Registry	保存实际允许调用的 Python 函数
Tool Executor	负责查找、验证和执行工具
Pydantic	验证模型生成的工具参数
"""