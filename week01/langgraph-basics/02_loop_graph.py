import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, START, END


# 1. 定义 State
class LoopState(TypedDict):
    n: int
    current: int
    total: int

    # 每次节点返回的新日志自动追加
    steps: Annotated[list[str], operator.add]


# 2. 累加节点
def accumulate_node(state: LoopState) -> dict:
    current = state["current"]
    total = state["total"] + current

    print(f"执行累加：{state['total']} + {current} = {total}")

    return {
        "total": total,
        "current": current + 1,
        "steps": [f"加上 {current}，当前结果为 {total}"],
    }


# 3. 条件路由
def route_after_accumulate(
    state: LoopState,
) -> Literal["continue", "finish"]:

    if state["current"] <= state["n"]:
        return "continue"

    return "finish"


# 4. 构建 Graph
builder = StateGraph(LoopState)

builder.add_node("accumulate", accumulate_node)

# 进入 Graph 时先检查是否需要累加
builder.add_conditional_edges(
    START,
    route_after_accumulate,
    {
        "continue": "accumulate",
        "finish": END,
    },
)

builder.add_conditional_edges(
    "accumulate",
    route_after_accumulate,
    {
        "continue": "accumulate",
        "finish": END,
    },
)

graph = builder.compile()


# 5. 执行
def main() -> None:
    initial_state: LoopState = {
        "n": 0,
        "current": 1,
        "total": 0,
        "steps": [],
    }

    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": 50},
    )

    print("\n最终状态：")
    print(final_state)


if __name__ == "__main__":
    main()
