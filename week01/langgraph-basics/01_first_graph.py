from typing import TypedDict

from langgraph.graph import StateGraph, START, END

# =========================
# 1. 定义 State
# =========================


class CalcState(TypedDict):
    a: int
    b: int
    factor: int
    need_multiply: bool
    result: int
    steps: list[str]


# =========================
# 2. 定义 Node
# =========================


def add_node(state: CalcState) -> dict:
    """执行加法"""

    result = state["a"] + state["b"]

    print(f"执行 add：{state['a']} + {state['b']} = {result}")

    return {
        "result": result,
        "steps": state["steps"] + ["执行 add"],
    }


def multiply_node(state: CalcState) -> dict:
    """执行乘法"""

    result = state["result"] * state["factor"]

    print(f"执行 multiply：" f"{state['result']} × {state['factor']} = {result}")

    return {
        "result": result,
        "steps": state["steps"] + ["执行 multiply"],
    }


# =========================
# 3. 条件路由
# =========================


def route_after_add(state: CalcState) -> str:
    """决定是否执行乘法"""

    if state["need_multiply"]:
        return "multiply"

    return "finish"


# =========================
# 4. 创建并连接 StateGraph
# =========================

builder = StateGraph(CalcState)

# 注册节点
builder.add_node("add", add_node)
builder.add_node("multiply", multiply_node)

# 从 START 进入 add
builder.add_edge(START, "add")

# add 后根据条件选择下一步
builder.add_conditional_edges(
    "add",
    route_after_add,
    {
        "multiply": "multiply",
        "finish": END,
    },
)

# multiply 执行完后结束
builder.add_edge("multiply", END)

# 编译图
graph = builder.compile()


# =========================
# 5. 执行
# =========================


def main() -> None:

    initial_state = {
        "a": 18,
        "b": 24,
        "factor": 3,
        "need_multiply": False,
        "result": 0,
        "steps": [],
    }

    final_state = graph.invoke(initial_state)

    print("\n最终状态：")
    print(final_state)


if __name__ == "__main__":
    main()
