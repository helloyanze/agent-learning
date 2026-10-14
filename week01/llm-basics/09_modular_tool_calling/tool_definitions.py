# 09
from pydantic import BaseModel, ConfigDict


# 1. 工具参数模型
class AddArgs(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    a: int
    b: int


# 2. 真正执行的 Python 函数
def add(a: int, b: int) -> int:
    return a + b


def multiply(a: int, b: int) -> int:
    return a * b


# 3. 工具注册表
TOOL_REGISTRY = {
    "add": add,
    "multiply": multiply,
}


# 4. 参数模型注册表
TOOL_ARGS_MODELS = {
    "add": AddArgs,
    "multiply": AddArgs,
}


# 5. 提供给 LLM 的工具描述
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "计算两个整数的和",
            "parameters": AddArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "计算两个整数的乘积",
            "parameters": AddArgs.model_json_schema(),
        },
    },
]
