import json
import os

from dotenv import load_dotenv
from openai import OpenAI

# =========================
# 1. 初始化 DeepSeek
# =========================
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if api_key is None:
    raise ValueError("没有找到 DEEPSEEK_API_KEY")


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


# ========================
# 2. 定义工具函数
# =======================
def add(a: int, b: int) -> int:
    return a + b


tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "计算两个整数的和",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "第一个整数",
                    },
                    "b": {
                        "type": "integer",
                        "description": "第二个整数",
                    },
                },
                "required": ["a", "b"],
            },
        },
    }
]

TOOL_REGISTRY = {
    "add": add,
}

# ========================
# 3. 调用工具函数
# ========================


def main() -> None:

    messages = [
        {
            "role": "user",
            "content": "帮我计算 18 加 24 等于多少？",
        }
    ]

    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        tools=tools,
    )

    message = response.choices[0].message

    print("模型返回的 Tool Calls：")
    print(message.tool_calls)

    # 1. 检查是否调用工具
    if not message.tool_calls:
        print("模型没有调用工具")
        print(message.content)
        return

    # 2. 提取工具名称和参数
    tool_call = message.tool_calls[0]

    tool_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    print(f"工具名称：{tool_name}")
    print(f"工具参数：{arguments}")

    # 3. 从 Registry 中查找工具
    tool_function = TOOL_REGISTRY.get(tool_name)

    if tool_function is None:
        raise ValueError(f"未知工具：{tool_name}")

    # 4. 执行工具
    result = tool_function(**arguments)

    print(f"工具执行结果：{result}")

    # 5. 把模型发起的工具调用加入历史
    messages.append(message)

    # 6. 把工具执行结果返回给模型
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result),
        }
    )

    # 7. 第二次调用 LLM
    final_response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        tools=tools,
    )

    print("模型最终回答：")
    print(final_response.choices[0].message.content)


if __name__ == "__main__":
    main()
