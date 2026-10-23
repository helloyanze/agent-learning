import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# 1. 定义 State
class MemoryState(TypedDict):
    increment: int
    total: int
    history: Annotated[list[str], operator.add]


# 2. 定义节点
def add_node(state: MemoryState) -> dict:
    increment = state["increment"]
    new_total = state["total"] + increment

    print(f"{state['total']} + {increment} = {new_total}")

    return {
        "total": new_total,
        "history": [f"加上 {increment}，得到 {new_total}"],
    }


# 3. 构建 Graph
builder = StateGraph(MemoryState)

builder.add_node("add", add_node)
builder.add_edge(START, "add")
builder.add_edge("add", END)

# 重点：启用 Checkpointer
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


def main() -> None:
    # 两个不同的会话
    config_a = {"configurable": {"thread_id": "user-a"}}
    config_b = {"configurable": {"thread_id": "user-b"}}

    # 4. 用户 A：第一次调用
    print("\n===== 用户 A：第一次调用 =====")

    result_a1 = graph.invoke(
        {
            "increment": 2,
            "total": 0,
            "history": [],
        },
        config=config_a,
    )

    print(result_a1)

    # 5. 用户 A：第二次调用
    print("\n===== 用户 A：第二次调用 =====")

    # 注意：只传入需要更新的 increment
    result_a2 = graph.invoke(
        {"increment": 3},
        config=config_a,
    )

    print(result_a2)

    # 6. 用户 B：独立会话
    print("\n===== 用户 B：第一次调用 =====")

    result_b1 = graph.invoke(
        {
            "increment": 10,
            "total": 0,
            "history": [],
        },
        config=config_b,
    )

    print(result_b1)

    # 7. 读取保存的状态
    print("\n===== 读取 Checkpoint =====")

    snapshot_a = graph.get_state(config_a)
    snapshot_b = graph.get_state(config_b)

    print("用户 A 状态：", snapshot_a.values)
    print("用户 B 状态：", snapshot_b.values)


if __name__ == "__main__":
    main()
