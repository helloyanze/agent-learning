import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from tool_definitions import TOOLS
from tool_executor import execute_tool

# =========================
# 1. 初始化 DeepSeek
# =========================

# .env 位于上一级 llm-basics 目录
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("没有找到 DEEPSEEK_API_KEY，请检查 .env 文件")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


# =========================
# 2. 定义 Agent System Prompt
# =========================

SYSTEM_PROMPT = """
你是一个计算助手。

你可以使用以下工具：
- add：计算两个整数的和
- multiply：计算两个整数的乘积

要求：
1. 遇到计算任务，优先使用工具，不要自行心算。
2. 如果计算存在前后依赖，需要先执行前一个工具。
3. 等待前一个工具返回结果后，再调用下一个工具。
4. 最终回答必须根据实际工具结果生成。
5. 如果工具执行失败，不要编造结果。
"""


# =========================
# 3. Agent Loop
# =========================


def run_agent(user_input: str, max_rounds: int = 5) -> str | None:

    # 保存对话历史
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    # Agent 最多执行 max_rounds 轮
    for round_num in range(1, max_rounds + 1):

        print(f"\n========== 第 {round_num} 轮 ==========")

        # 第一步：调用 LLM
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        message = choice.message

        # 避免处理截断的工具调用
        if choice.finish_reason == "length":
            print("模型输出被截断，停止执行")
            return None

        # 把模型消息加入对话历史
        messages.append(message)

        # 第二步：判断模型是否还需要工具
        if not message.tool_calls:

            final_answer = message.content

            if not final_answer:
                print("模型未返回有效回答")
                return None

            print("\nAgent 最终回答：")
            print(final_answer)

            return final_answer

        # 第三步：执行这一轮请求的所有工具
        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name
            arguments_json = tool_call.function.arguments

            print(f"\n工具名称：{tool_name}")
            print(f"工具参数：{arguments_json}")

            # 第四步：通过 Tool Executor 执行
            try:
                result = execute_tool(
                    tool_name=tool_name,
                    arguments_json=arguments_json,
                )

                print(f"工具执行成功：{result}")

                tool_result = result

            except (ValueError, ValidationError) as e:

                print(f"工具执行失败：{e}")

                tool_result = f"工具执行失败：{e}"

            # 第五步：把工具结果加入对话历史
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

        # 当前轮结束，进入下一轮
        # LLM 会读取前面的工具结果并决定下一步

    print(f"\n已达到最大轮数 {max_rounds}，Agent 停止执行")
    return None


# =========================
# 4. 主程序
# =========================


def main() -> None:

    user_input = """
    请帮我计算 (18 + 24) × 2。

    要求：
    1. 先调用 add 工具计算 18 + 24。
    2. 得到 add 的实际结果之后，再调用 multiply。
    3. 根据两个工具的执行结果回答。
    """

    result = run_agent(
        user_input=user_input,
        max_rounds=5,
    )

    if result is None:
        print("本次任务未成功完成")


# =========================
# 5. 启动程序
# =========================

if __name__ == "__main__":
    main()
