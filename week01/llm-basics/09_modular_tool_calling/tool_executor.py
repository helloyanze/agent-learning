# 09
from tool_definitions import TOOL_REGISTRY, TOOL_ARGS_MODELS


def execute_tool(tool_name: str, arguments_json: str) -> str:

    # 1. 查找对应工具
    tool_function = TOOL_REGISTRY.get(tool_name)

    # 2. 查找工具参数模型
    args_model = TOOL_ARGS_MODELS.get(tool_name)

    # 3. 判断工具是否存在
    if tool_function is None or args_model is None:
        raise ValueError(f"未知工具：{tool_name}")

    # 4. 解析 JSON 并验证参数
    validated_args = args_model.model_validate_json(arguments_json)

    # 5. 执行工具
    result = tool_function(**validated_args.model_dump())

    # 6. 返回工具结果
    return str(result)
