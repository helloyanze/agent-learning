import operator
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


class State(TypedDict):
    increment: int
    total: int
    history: Annotated[list[str], operator.add]


def add_node(state: State) -> dict:
    total = state["total"] + state["increment"]

    return {
        "total": total,
        "history": [f"累计结果：{total}"],
    }


builder = StateGraph(State)
builder.add_node("add", add_node)
builder.add_edge(START, "add")
builder.add_edge("add", END)


def main():
    db_path = Path(__file__).with_name("task13_checkpoints.sqlite")

    config = {"configurable": {"thread_id": "demo-user"}}

    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:

        graph = builder.compile(checkpointer=checkpointer)

        # 查询是否已有保存的状态
        old_state = graph.get_state(config).values

        if old_state:
            print("发现历史状态：", old_state)
            new_input = {"increment": 3}
        else:
            print("首次运行，初始化状态")
            new_input = {
                "increment": 2,
                "total": 0,
                "history": [],
            }

        result = graph.invoke(
            new_input,
            config=config,
        )

        print("本次运行结果：", result)


if __name__ == "__main__":
    main()
