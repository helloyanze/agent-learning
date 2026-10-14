from dataclasses import asdict

from agent_state import (
    AgentState,
    start_round,
    start_tool,
    complete,
)


def main() -> None:

    state = AgentState(
        max_rounds=3,
        max_tool_calls=2,
    )

    print("初始状态：")
    print(asdict(state))

    print("\n开始第一轮")
    print(start_round(state))

    print("\n执行 add")
    print(start_tool(state, "add"))

    print("\n开始第二轮")
    print(start_round(state))

    print("\n执行 multiply")
    print(start_tool(state, "multiply"))

    print("\n尝试第三次工具调用")
    print(start_tool(state, "add"))

    print("\n最终状态：")
    print(asdict(state))


if __name__ == "__main__":
    main()
